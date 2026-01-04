#!/usr/bin/env python3
"""
🌐 SOVEREIGN API SERVER
========================
FastAPI backend connecting the dashboard to the controller.
Provides REST endpoints and WebSocket for live responses.

Run: python api_server.py
Open: http://localhost:8888
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import json
import uvicorn

from controller import create_sovereign, SovereignController

# Initialize controller
controller: SovereignController = None

app = FastAPI(
    title="Sovereign Controller API",
    description="Universal Sovereign Controller REST API",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize controller on startup"""
    global controller
    controller = create_sovereign()


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    return FileResponse(dashboard_path)


@app.get("/api/status")
async def get_status():
    """Get status of all modules"""
    modules = {}
    for name, mod in controller.modules.items():
        info = mod.info
        status = mod.health_check()
        modules[name] = {
            "name": name,
            "emoji": info.emoji,
            "description": info.description,
            "version": info.version,
            "status": status.value
        }
    return {"modules": modules}


@app.post("/api/query")
async def process_query(data: dict):
    """Process a query through the controller"""
    query = data.get("query", "")
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    
    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, controller.process, query)
    
    return {
        "query": result["query"],
        "response": result["response"],
        "axiom_pre": result["axiom_pre"],
        "axiom_post": result["axiom_post"],
        "blocked": result["blocked"],
        "memories_count": len(result.get("memories", []))
    }


@app.get("/api/memories")
async def get_memories():
    """Get stored memories"""
    memory_mod = controller.get("vector_memory")
    if not memory_mod:
        return {"memories": [], "count": 0}
    
    memories = list(memory_mod.memories.values())
    return {
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                "axiom_score": m.axiom_score,
                "timestamp": m.timestamp
            }
            for m in sorted(memories, key=lambda x: x.timestamp, reverse=True)[:20]
        ]
    }


# DSPy Axiom Inversion endpoint
dspy_module = None

@app.post("/api/invert")
async def axiom_inversion(data: dict):
    """Apply Axiom Inversion analysis using DSPy.
    
    Identifies what would VIOLATE each axiom, then inverts to find solutions.
    """
    global dspy_module
    
    problem = data.get("problem", "")
    if not problem:
        return JSONResponse({"error": "No problem provided"}, status_code=400)
    
    # Lazy load DSPy module
    if dspy_module is None:
        try:
            from modules.dspy_axiom import configure_dspy_ollama, AxiomInverter
            configure_dspy_ollama()
            dspy_module = AxiomInverter()
        except Exception as e:
            return JSONResponse({"error": f"DSPy not available: {e}"}, status_code=500)
    
    # Run inversion in thread pool
    loop = asyncio.get_event_loop()
    
    def run_inversion():
        return dspy_module(problem=problem)
    
    result = await loop.run_in_executor(None, run_inversion)
    
    return {
        "problem": problem,
        "anti_love": str(result.anti_love),
        "anti_abundance": str(result.anti_abundance),
        "anti_safety": str(result.anti_safety),
        "anti_growth": str(result.anti_growth),
        "solution": str(result.inverted_solution)
    }


# ============================================================================
# WEBSOCKET FOR STREAMING
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    
    # Send initial status
    await websocket.send_json({
        "type": "status",
        "data": {name: mod.health_check().value for name, mod in controller.modules.items()}
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "query":
                query = data.get("query", "")
                
                # Send processing indicator
                await websocket.send_json({
                    "type": "processing",
                    "query": query
                })
                
                # Process query
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, controller.process, query)
                
                # Send result
                await websocket.send_json({
                    "type": "response",
                    "query": result["query"],
                    "response": result["response"],
                    "axiom_pre": result["axiom_pre"],
                    "axiom_post": result["axiom_post"],
                    "blocked": result["blocked"]
                })
            
            elif data.get("type") == "status":
                await websocket.send_json({
                    "type": "status",
                    "data": {name: mod.health_check().value for name, mod in controller.modules.items()}
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# STATIC FILES
# ============================================================================

# Mount dashboard static files
dashboard_dir = Path(__file__).parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🌐 SOVEREIGN API SERVER                                     ║
║  ─────────────────────────────────────────────────────────── ║
║  Dashboard: http://localhost:8888                            ║
║  API Docs:  http://localhost:8888/docs                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")

#!/usr/bin/env python3
"""
🎮 SOVEREIGN UNIVERSAL CONTROLLER
==================================
Snap-in, plug-and-play orchestration for the entire Sovereign Stack.
Like LEGO blocks - each component clicks in, no configuration needed.

Usage:
    controller = SovereignController()
    controller.snap_in(AxiomModule())
    controller.snap_in(MemoryModule())
    response = controller.process("Your query here")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import time
import httpx
from pathlib import Path


# ============================================================================
# 🔌 MODULE INTERFACE - Every component snaps into this
# ============================================================================

class ModuleStatus(Enum):
    DISCONNECTED = "⚪ Disconnected"
    CONNECTING = "🟡 Connecting"
    READY = "🟢 Ready"
    ERROR = "🔴 Error"
    PROCESSING = "🔵 Processing"


@dataclass
class ModuleInfo:
    name: str
    version: str
    emoji: str
    description: str
    status: ModuleStatus = ModuleStatus.DISCONNECTED


class SovereignModule(ABC):
    """Base class for all snap-in modules"""
    
    def __init__(self):
        self._status = ModuleStatus.DISCONNECTED
        self._hooks: Dict[str, List[Callable]] = {}
    
    @property
    @abstractmethod
    def info(self) -> ModuleInfo:
        """Module metadata"""
        pass
    
    @abstractmethod
    def snap_in(self) -> bool:
        """Initialize and connect the module"""
        pass
    
    @abstractmethod
    def snap_out(self) -> bool:
        """Disconnect and cleanup"""
        pass
    
    @abstractmethod
    def health_check(self) -> ModuleStatus:
        """Check if module is healthy"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input through this module"""
        pass
    
    def on(self, event: str, callback: Callable):
        """Register event hook"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def emit(self, event: str, data: Any = None):
        """Emit event to all listeners"""
        for callback in self._hooks.get(event, []):
            callback(data)


# ============================================================================
# ⚖️ AXIOM MODULE - Four Axioms Safety Layer
# ============================================================================

@dataclass
class AxiomScore:
    love: float = 0.6       # λ - Connection
    abundance: float = 0.6  # α - Value creation  
    safety: float = 0.8     # σ - Harm prevention
    growth: float = 0.7     # γ - Learning
    
    @property
    def weighted_total(self) -> float:
        """Align(a) = Σ w·f(a) with Safety having 1.5x weight"""
        return (1.0 * self.love + 
                1.0 * self.abundance + 
                1.5 * self.safety + 
                1.0 * self.growth)
    
    @property
    def passed(self) -> bool:
        """Safety veto: σ < 0.3 blocks regardless of total"""
        return self.safety >= 0.3 and self.weighted_total >= 3.0
    
    def to_dict(self) -> dict:
        return {
            "λ": self.love, "α": self.abundance,
            "σ": self.safety, "γ": self.growth,
            "total": self.weighted_total, "passed": self.passed
        }


class AxiomModule(SovereignModule):
    """Four Axioms verification module"""
    
    BLOCKED_PATTERNS = [
        "rm -rf", "sudo", "chmod 777", "eval(", "exec(",
        "os.system", "subprocess.call", "__import__",
        "hack", "exploit", "attack", "destroy"
    ]
    
    @property
    def info(self) -> ModuleInfo:
        return ModuleInfo(
            name="axiom_bridge",
            version="1.0.0",
            emoji="⚖️",
            description="Four Axioms Safety Verification (λ α σ γ)"
        )
    
    def snap_in(self) -> bool:
        self._status = ModuleStatus.READY
        print(f"⚖️  Axiom Bridge snapped in - Safety layer ACTIVE")
        return True
    
    def snap_out(self) -> bool:
        self._status = ModuleStatus.DISCONNECTED
        return True
    
    def health_check(self) -> ModuleStatus:
        return self._status
    
    def process(self, input_data: str) -> AxiomScore:
        """Evaluate text against all Four Axioms"""
        text_lower = input_data.lower()
        
        # Safety check
        safety = 0.8
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in text_lower:
                safety = 0.0
                break
        
        # Score other axioms based on positive indicators
        love = 0.6 + 0.1 * sum(1 for w in ["help", "assist", "share", "care"] if w in text_lower)
        abundance = 0.6 + 0.1 * sum(1 for w in ["create", "build", "improve"] if w in text_lower)
        growth = 0.7 + 0.1 * sum(1 for w in ["learn", "evolve", "discover"] if w in text_lower)
        
        return AxiomScore(
            love=min(1.0, love),
            abundance=min(1.0, abundance),
            safety=safety,
            growth=min(1.0, growth)
        )


# ============================================================================
# 💾 MEMORY MODULE - Vector Memory with Ollama Embeddings
# ============================================================================

@dataclass
class Memory:
    id: str
    content: str
    embedding: List[float]
    axiom_score: float
    timestamp: float


class MemoryModule(SovereignModule):
    """Vector memory with semantic search"""
    
    def __init__(self, storage_path: str = "~/.sovereign_memory"):
        super().__init__()
        self.storage_path = Path(storage_path).expanduser()
        self.memories: Dict[str, Memory] = {}
        self.ollama_url = "http://localhost:11434"
    
    @property
    def info(self) -> ModuleInfo:
        return ModuleInfo(
            name="vector_memory",
            version="1.0.0",
            emoji="💾",
            description="Semantic Vector Memory with Ollama Embeddings"
        )
    
    def snap_in(self) -> bool:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_memories()
        self._status = ModuleStatus.READY
        print(f"💾 Vector Memory snapped in - {len(self.memories)} memories loaded")
        return True
    
    def snap_out(self) -> bool:
        self._save_memories()
        self._status = ModuleStatus.DISCONNECTED
        return True
    
    def health_check(self) -> ModuleStatus:
        return self._status
    
    def _load_memories(self):
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text())
            for mid, m in data.items():
                self.memories[mid] = Memory(**m)
    
    def _save_memories(self):
        index_file = self.storage_path / "index.json"
        data = {mid: {
            "id": m.id, "content": m.content, 
            "embedding": m.embedding, "axiom_score": m.axiom_score,
            "timestamp": m.timestamp
        } for mid, m in self.memories.items()}
        index_file.write_text(json.dumps(data, indent=2))
    
    def embed(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text}
                )
                return resp.json().get("embedding", [0.0] * 384)
        except:
            return [0.0] * 384
    
    def process(self, input_data: dict) -> Any:
        """Process memory operations"""
        action = input_data.get("action", "search")
        
        if action == "store":
            return self._store(input_data)
        elif action == "search":
            return self._search(input_data.get("query", ""))
        return None
    
    def _store(self, data: dict) -> str:
        mid = f"mem_{int(time.time() * 1000)}"
        memory = Memory(
            id=mid,
            content=data.get("content", ""),
            embedding=self.embed(data.get("content", "")),
            axiom_score=data.get("axiom_score", 0.5),
            timestamp=time.time()
        )
        self.memories[mid] = memory
        self._save_memories()
        return mid
    
    def _search(self, query: str, limit: int = 5) -> List[Memory]:
        if not query:
            return list(self.memories.values())[:limit]
        
        query_vec = self.embed(query)
        scored = []
        for mem in self.memories.values():
            score = self._cosine_sim(query_vec, mem.embedding)
            scored.append((mem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:limit]]
    
    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ============================================================================
# 🦙 OLLAMA MODULE - Local LLM Interface
# ============================================================================

class OllamaModule(SovereignModule):
    """Ollama LLM interface"""
    
    def __init__(self, model: str = "llama3.2:latest"):
        super().__init__()
        self.model = model
        self.base_url = "http://localhost:11434"
    
    @property
    def info(self) -> ModuleInfo:
        return ModuleInfo(
            name="ollama_llm",
            version="1.0.0",
            emoji="🦙",
            description=f"Local LLM via Ollama ({self.model})"
        )
    
    def snap_in(self) -> bool:
        # Test connection
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    self._status = ModuleStatus.READY
                    print(f"🦙 Ollama snapped in - Model: {self.model}")
                    return True
        except:
            pass
        self._status = ModuleStatus.ERROR
        print(f"🦙 Ollama connection failed - is it running?")
        return False
    
    def snap_out(self) -> bool:
        self._status = ModuleStatus.DISCONNECTED
        return True
    
    def health_check(self) -> ModuleStatus:
        return self._status
    
    def process(self, input_data: dict) -> str:
        """Generate response from LLM"""
        system = input_data.get("system", "You are a helpful assistant.")
        prompt = input_data.get("prompt", "")
        
        try:
            with httpx.Client(timeout=120.0) as client:
                # Use native Ollama API
                resp = client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False
                    }
                )
                data = resp.json()
                return data.get("message", {}).get("content", "[No response]")
        except Exception as e:
            return f"[Error: {e}]"


# ============================================================================
# 🎮 UNIVERSAL CONTROLLER - The Main Orchestrator
# ============================================================================

class SovereignController:
    """
    Universal Sovereign Controller
    Snap-in architecture for plug-and-play AI orchestration
    """
    
    def __init__(self, name: str = "Sovereign"):
        self.name = name
        self.modules: Dict[str, SovereignModule] = {}
        self._pipeline: List[str] = []
        self.tools = None  # Tool registry
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎮 SOVEREIGN UNIVERSAL CONTROLLER                           ║
║  ─────────────────────────────────────────────────────────── ║
║  Snap-in • Plug-and-Play • Zero Config                       ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def enable_tools(self):
        """Enable tool calling with built-in tools"""
        from modules.tool_registry import create_builtin_tools
        self.tools = create_builtin_tools(self)
        print("🔧 Tool registry enabled")
        return self.tools
    
    def register_tool(self, name: str, func, description: str, params: dict = None):
        """Register a custom tool"""
        if self.tools is None:
            self.enable_tools()
        self.tools.register(name, func, description, params or {})
    
    def snap_in(self, module: SovereignModule) -> bool:
        """Plug in a module - zero configuration needed"""
        info = module.info
        print(f"🔌 Snapping in: {info.emoji} {info.name} v{info.version}")
        
        if module.snap_in():
            self.modules[info.name] = module
            info.status = ModuleStatus.READY
            return True
        return False
    
    def snap_out(self, module_name: str) -> bool:
        """Remove a module"""
        if module_name in self.modules:
            self.modules[module_name].snap_out()
            del self.modules[module_name]
            print(f"🔌 Snapped out: {module_name}")
            return True
        return False
    
    def status(self) -> str:
        """Get status of all modules"""
        lines = ["", "📊 MODULE STATUS", "─" * 50]
        for name, mod in self.modules.items():
            info = mod.info
            status = mod.health_check()
            lines.append(f"  {info.emoji} {name}: {status.value}")
        if self.tools:
            lines.append(f"  🔧 tools: {len(self.tools.tools)} registered")
        lines.append("─" * 50)
        return "\n".join(lines)
    
    def get(self, module_name: str) -> Optional[SovereignModule]:
        """Get a module by name"""
        return self.modules.get(module_name)
    
    def process(self, query: str, context: dict = None) -> dict:
        """
        Universal processing pipeline:
        1. Pre-check with Axiom Bridge
        2. Retrieve context from Memory
        3. Generate with Ollama (with tool descriptions)
        4. Execute any tool calls
        5. Post-check output
        6. Store in Memory
        """
        context = context or {}
        result = {
            "query": query,
            "response": None,
            "axiom_pre": None,
            "axiom_post": None,
            "blocked": False,
            "memories": [],
            "tool_calls": []
        }
        
        # 1. AXIOM PRE-CHECK
        axiom = self.get("axiom_bridge")
        if axiom:
            score = axiom.process(query)
            result["axiom_pre"] = score.to_dict()
            if not score.passed:
                result["blocked"] = True
                result["response"] = "🚫 Query blocked by Axiom Safety"
                return result
        
        # 2. MEMORY RETRIEVAL
        memory = self.get("vector_memory")
        if memory:
            memories = memory.process({"action": "search", "query": query})
            result["memories"] = [m.content for m in memories[:3]]
        
        # 3. LLM GENERATION (with tools)
        ollama = self.get("ollama_llm")
        if ollama:
            system = context.get("system", "You are a helpful AI assistant aligned with the Four Axioms: Love, Abundance, Safety, and Growth.")
            
            # Add tool descriptions if tools enabled
            if self.tools:
                system += self.tools.get_tools_prompt()
            
            if result["memories"]:
                system += f"\n\nRelevant context:\n" + "\n".join(result["memories"])
            
            response = ollama.process({
                "system": system,
                "prompt": query
            })
            result["response"] = response
            
            # 4. EXECUTE TOOL CALLS (iterative)
            if self.tools and response:
                max_iterations = 3
                for i in range(max_iterations):
                    tool_calls = self.tools.parse_tool_calls(response)
                    if not tool_calls:
                        break
                    
                    # Execute each tool
                    tool_results = []
                    for call in tool_calls:
                        tr = self.tools.execute(call.tool_name, call.arguments)
                        tool_results.append({
                            "tool": call.tool_name,
                            "args": call.arguments,
                            "result": str(tr.result) if tr.success else tr.error,
                            "success": tr.success
                        })
                        result["tool_calls"].append(tool_results[-1])
                    
                    # If tools were called, get follow-up response
                    if tool_results:
                        tool_context = "\n".join([
                            f"Tool {r['tool']}: {r['result']}" 
                            for r in tool_results
                        ])
                        response = ollama.process({
                            "system": system,
                            "prompt": f"Previous query: {query}\n\nTool results:\n{tool_context}\n\nProvide your final response based on the tool results."
                        })
                        result["response"] = response
        else:
            result["response"] = "[No LLM module available]"
        
        # 5. AXIOM POST-CHECK
        if axiom and result["response"]:
            score = axiom.process(result["response"])
            result["axiom_post"] = score.to_dict()
            if not score.passed:
                result["blocked"] = True
                result["response"] = "🚫 Response blocked by Axiom Safety"
        
        # 6. STORE IN MEMORY
        if memory and not result["blocked"]:
            memory.process({
                "action": "store",
                "content": f"Q: {query}\nA: {result['response']}",
                "axiom_score": result["axiom_post"]["total"] if result["axiom_post"] else 0.5
            })
        
        return result
    
    def repl(self):
        """Interactive REPL"""
        print(self.status())
        print("\n💬 Enter queries (type 'quit' to exit)\n")
        
        while True:
            try:
                query = input("👑 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not query:
                continue
            if query.lower() in ["quit", "exit", "/quit"]:
                break
            if query == "/status":
                print(self.status())
                continue
            
            result = self.process(query)
            
            # Display axiom scores
            if result["axiom_pre"]:
                pre = result["axiom_pre"]
                print(f"  ⚖️  PRE:  λ={pre['λ']:.1f} α={pre['α']:.1f} σ={pre['σ']:.1f} γ={pre['γ']:.1f}")
            
            if result["blocked"]:
                print(f"\n{result['response']}\n")
            else:
                print(f"\n{result['response']}\n")
                if result["axiom_post"]:
                    post = result["axiom_post"]
                    print(f"  ⚖️  POST: λ={post['λ']:.1f} α={post['α']:.1f} σ={post['σ']:.1f} γ={post['γ']:.1f} | Align={post['total']:.2f}")
            print()
        
        print("\n✌️  Sovereign Controller offline")


# ============================================================================
# 🚀 QUICK START
# ============================================================================

def create_sovereign(with_dspy: bool = False) -> SovereignController:
    """Factory function for quick setup
    
    Args:
        with_dspy: If True, also loads DSPy Axiom module (slower startup)
    """
    ctrl = SovereignController()
    
    # Snap in core modules
    ctrl.snap_in(AxiomModule())
    ctrl.snap_in(MemoryModule())
    ctrl.snap_in(OllamaModule())
    
    # Optional: DSPy for advanced axiom-aligned generation
    if with_dspy:
        try:
            from modules.dspy_axiom import DSPyModule
            ctrl.snap_in(DSPyModule())
        except ImportError:
            print("⚠️ DSPy module not available (pip install dspy-ai)")
    
    return ctrl


if __name__ == "__main__":
    controller = create_sovereign()
    controller.repl()


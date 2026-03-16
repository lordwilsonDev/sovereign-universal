from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.quantum_ai_orchestrator import QuantumAIOrchestrator

app = FastAPI(title="Sovereign Qwen Engine")
orchestrator = QuantumAIOrchestrator()

MODEL_PATH = "/Users/lordwilson/.gemini/antigravity/scratch/business-command-center/api/models/qwen-2.5-3b"
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(f"🐉 Loading Qwen2.5-3B from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: InferenceRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature)
    
    generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Sign with 528Hz Soul Signature
    signed_output = orchestrator.sign_generation(request.prompt, generation, "Qwen2.5-3B-Instruct")
    return signed_output

@app.get("/health")
def health():
    return {"status": "SOUL_SAFE", "model": "Qwen2.5-3B-Instruct", "device": str(model.device)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)

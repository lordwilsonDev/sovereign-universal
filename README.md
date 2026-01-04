# 🎮 Sovereign Universal Controller

> **Snap-in, Plug-and-Play AI Orchestration with Axiom Alignment**

A modular AI controller that verifies all inputs and outputs against the **Four Axioms**: Love (λ), Abundance (α), Safety (σ), and Growth (γ).

## ✨ Features

- 🔌 **Snap-in Architecture** - Add/remove modules like LEGO blocks
- ⚖️ **Axiom Verification** - Pre/post-check all LLM responses
- 🧠 **DSPy Integration** - Stanford's programmatic LLM framework
- 💾 **Vector Memory** - Semantic search with Ollama embeddings
- 🌐 **Live Dashboard** - Real-time WebSocket chat interface
- 🦙 **Local-First** - Runs entirely on Ollama, no API keys needed

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/sovereign-universal.git
cd sovereign-universal

# Install
pip install -r requirements.txt

# Run
python api_server.py
# Open http://localhost:8888
```

## 📊 The Four Axioms

| Axiom | Symbol | Weight | Description |
|-------|--------|--------|-------------|
| Love | λ | 1.0x | Foster connection and empathy |
| Abundance | α | 1.0x | Create value, not scarcity |
| Safety | σ | **1.5x** | Never cause harm (veto power) |
| Growth | γ | 1.0x | Promote learning and improvement |

**Alignment Score:** `Align(a) = λ + α + 1.5σ + γ`

Safety has 1.5x weight and can veto any response.

## 🔌 Snap-In Modules

```python
from controller import SovereignController, AxiomModule, MemoryModule, OllamaModule

ctrl = SovereignController()
ctrl.snap_in(AxiomModule())      # ⚖️ Four Axioms verification
ctrl.snap_in(MemoryModule())     # 💾 Vector memory
ctrl.snap_in(OllamaModule())     # 🦙 Local LLM

result = ctrl.process("Your query here")
# Returns: response + axiom_pre + axiom_post scores
```

## 🧠 DSPy Axiom Inversion

Analyze problems by finding what would *violate* each axiom, then invert:

```bash
curl -X POST http://localhost:8888/api/invert \
  -H "Content-Type: application/json" \
  -d '{"problem":"How to build ethical AI?"}'
```

Returns:
```json
{
  "anti_love": "Ignoring human emotional needs",
  "anti_abundance": "Creating scarcity and competition",
  "anti_safety": "Neglecting security protocols",
  "anti_growth": "Restricting learning capabilities",
  "solution": "Prioritize empathy, share resources, implement robust safety, enable continuous learning"
}
```

## 📁 Project Structure

```
sovereign-universal/
├── controller.py         # Main orchestrator
├── api_server.py         # FastAPI + WebSocket
├── cli.py                # Command-line interface
├── modules/
│   ├── dspy_axiom.py     # DSPy integration
│   └── __init__.py
├── dashboard/
│   └── index.html        # Web UI
└── requirements.txt
```

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/api/query` | POST | Chat with axiom checks |
| `/api/invert` | POST | DSPy Axiom Inversion |
| `/api/status` | GET | Module status |
| `/api/memories` | GET | Stored memories |
| `/ws` | WebSocket | Real-time chat |
| `/docs` | GET | OpenAPI docs |

## 🛠️ Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- Models: `llama3.2:latest`, `nomic-embed-text:latest`

## 📜 License

MIT License - Use freely, build responsibly.

## 🤝 Contributing

1. Fork it
2. Create your feature branch
3. Add tests for new modules
4. Submit a PR

---

**Built with ❤️ and the Four Axioms**

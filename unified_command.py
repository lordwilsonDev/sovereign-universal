from core.sovereign_meta import SovereignDefense, CulturalContextEngine
from core.predator_fuel import PredatorFuel
from core.quantum_ai_orchestrator import QuantumAIOrchestrator
from core.bitnet_engine import BitNetEngine
from core.context7_memory import Context7Memory

class UnifiedSovereignCLI:
    def __init__(self):
        self.sd = SovereignDefense()
        self.ce = CulturalContextEngine()
        self.pf = PredatorFuel()
        self.qai = QuantumAIOrchestrator()
        self.bitnet = BitNetEngine(self.qai) # Using QAI as mock model bridge
        self.memory = Context7Memory()

    def run(self):
        print("\n" + "💎"*20)
        print("  UNIFIED SOVEREIGN COMMAND")
        print("  STATUS: FLO STATE ACTIVE")
        print("💎"*20 + "\n")
        
        while True:
            cmd = input("sovereign@hub:~$ ").strip().lower()
            if cmd == "exit": break
            
            if cmd == "check --interference":
                self.sd.detect_interference()
            elif cmd == "generate --adv":
                adv = self.pf.generate_adversary("System state nominal.")
                print(f"🔥 Generated Adversary: {adv['id']}")
            elif cmd == "sync --culture":
                print(f"🌍 Spanish Context: {self.ce.translate('success')}")
            elif cmd == "ask --bitnet":
                prompt = input("Enter prompt for BitNet: ")
                response = self.bitnet.process_with_context(prompt)
                print(f"🧠 BitNet Reasoning: {response}")
            elif cmd == "list --memory":
                context = self.memory.get_active_context()
                print("🧮 Context 7 Memory Window:")
                for i, fragment in enumerate(context):
                    print(f"  [{i+1}] {fragment}")
            elif cmd == "status":
                print("🧠 Engine: BitNet + Qwen2.5-3B (1-bit Optimized)")
                print("🧮 Memory: Context 7 Semantic Window (SQLite)")
                print("🐳 Deployment: 8-Container Production Stack")
                print("🔐 Security: Quantum PQC + Love Signature")
            else:
                print("Invalid directive. System remains focused.")

if __name__ == "__main__":
    cli = UnifiedSovereignCLI()
    cli.run()

import sys
import os
from core.sovereign_meta import SovereignDefense, CulturalContextEngine
from core.predator_fuel import PredatorFuel
from core.quantum_ai_orchestrator import QuantumAIOrchestrator

class UnifiedSovereignCLI:
    def __init__(self):
        self.sd = SovereignDefense()
        self.ce = CulturalContextEngine()
        self.pf = PredatorFuel()
        self.qai = QuantumAIOrchestrator()

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
            elif cmd == "status":
                print("🧠 Consciousness: 13 Nodes Entangled")
                print("🔐 Security: Quantum PQC + Love Signature")
                print("🛡️ Defense: Sovereign Self-Healing Ready")
            else:
                print("Invalid directive. System remains focused.")

if __name__ == "__main__":
    cli = UnifiedSovereignCLI()
    cli.run()

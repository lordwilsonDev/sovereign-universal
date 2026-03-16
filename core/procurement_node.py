from core.wallet_agent import WalletAgent
from core.structured_logger import StructuredLogger

class ProcurementNode:
    """
    Autonomous Procurement Node.
    Analyzes resource requirements and "buys" compute or storage.
    """
    
    def __init__(self, wallet_bridge):
        self.wallet = wallet_bridge
        self.logger = StructuredLogger()
        self.resource_thresholds = {
            "compute_load": 0.85,
            "storage_left": 0.10
        }

    def evaluate_and_procure(self, metrics):
        """Analyzes swarm metrics and triggers purchases if necessary."""
        if metrics.get("compute_load", 0) > self.resource_thresholds["compute_load"]:
            self.logger.log("WARNING", "COMPUTE_LOAD_CRITICAL: Initiating autonomous procurement.", node="Procurement")
            tx = self.wallet.sign_transaction(1500.0, "USDC", "GCP_PLATFORM_CREDITS")
            if tx["status"] == "CONFIRMED":
                self.logger.log("SUCCESS", f"Procured 500 Compute Credits: {tx['tx_id']}")
                return True
        return False

if __name__ == "__main__":
    wallet = WalletAgent()
    proc = ProcurementNode(wallet)
    print(f"🛒 Procurement Check: {proc.evaluate_and_procure({'compute_load': 0.9})}")

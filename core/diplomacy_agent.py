import time

class DiplomacyAgent:
    """
    Sovereign Diplomacy Agent.
    Negotiates with other swarm instances and maintains the Universal Trust Ledger.
    """
    def __init__(self, did_manager):
        self.did = did_manager.did
        self.trust_ledger = {} # peer_did -> trust_score
        self.negotiation_history = []

    def negotiate_exchange(self, peer_did, resource_type, price_usdc):
        """Evaluates an exchange request from another swarm instance."""
        current_trust = self.trust_ledger.get(peer_did, 0.5) # Default trust 0.5
        
        if current_trust < 0.3:
            return {"status": "REJECTED", "reason": "LOW_TRUST_THRESHOLD"}
            
        print(f"🤝 DIPLOMACY: Negotiating {resource_type} with {peer_did} at ${price_usdc}")
        
        # Simple heuristic: accept if trust is high or price is reasonable
        if current_trust > 0.8 or price_usdc < 100:
            self.trust_ledger[peer_did] = min(1.0, current_trust + 0.05)
            return {"status": "ACCEPTED", "exchange_id": f"EX-{int(time.time())}"}
        
        return {"status": "COUNTER_OFFER", "suggested_price": price_usdc * 0.9}

    def audit_peer(self, peer_did, result_success):
        """Updates trust based on exchange performance."""
        current_trust = self.trust_ledger.get(peer_did, 0.5)
        if result_success:
            self.trust_ledger[peer_did] = min(1.0, current_trust + 0.1)
        else:
            self.trust_ledger[peer_did] = max(0.0, current_trust - 0.3)
        print(f"🛡️  AUDIT: {peer_did} Trust updated to {self.trust_ledger[peer_did]:.2f}")

if __name__ == "__main__":
    from core.sovereign_did import SovereignDID
    did_m = SovereignDID("SIMULATED_SOUL_BETA")
    da = DiplomacyAgent(did_m)
    
    peer = "did:sov:alpha_instance"
    print(f"🤝 Negotiation: {da.negotiate_exchange(peer, 'compute_cycles', 50)}")
    da.audit_peer(peer, True)

from core.quantum_crypto import QuantumTrinity

class QuantumGateway:
    """
    PQC-Hardened Sovereign API Gateway.
    Filters all incoming signals through a Quantum Trinity proxy.
    """
    
    def __init__(self):
        self.trinity = QuantumTrinity()
        self.blocked_ips = []

    def proxy_request(self, payload, signature, origin_ip):
        """Verifies PQC signature before allowing the signal to enter the swarm."""
        if origin_ip in self.blocked_ips:
            return {"status": "BLOCKED", "reason": "IP_BLACKLISTED"}
            
        if self.trinity.verify_soul_signature(payload, signature):
            return {"status": "AUTHORIZED", "payload": payload}
        else:
            self.blocked_ips.append(origin_ip)
            return {"status": "REJECTED", "reason": "INVALID_PQC_SIGNATURE"}

if __name__ == "__main__":
    qg = QuantumGateway()
    # Mock auth
    print(f"🔐 Gateway Authorization: {qg.proxy_request('GREETING', 'MOCK_SIG', '127.0.0.1')}")

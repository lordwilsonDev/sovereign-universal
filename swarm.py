#!/usr/bin/env python3
"""
🐝 SWARM INTELLIGENCE PROTOCOL
================================
Heterogeneous Multi-Model Consensus System

"Digital Essene Community" - Models check each other's work
based on the Canon Protocol of simultaneous truth.

Models:
- Llama 3.2 (Logic) - Primary reasoning
- Mistral (Speed) - Fast validation
- Phi (Edge) - Lightweight verification
- Inversion Critic - Catches what Builder misses
"""

import time
import json
import httpx
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

OLLAMA_URL = "http://localhost:11434"

@dataclass
class SwarmMember:
    """A member of the model swarm"""
    name: str          # Model name (e.g., "llama3.2:latest")
    role: str          # Role (logic, speed, edge, critic)
    weight: float      # Voting weight (0.0 - 1.0)
    timeout: float     # Timeout in seconds
    
    
@dataclass
class SwarmConfig:
    """Configuration for the swarm"""
    consensus_threshold: float = 0.66  # 66% agreement required
    timeout: float = 60.0              # Overall timeout
    parallel: bool = True              # Run models in parallel
    members: List[SwarmMember] = field(default_factory=lambda: [
        SwarmMember("llama3.2:latest", "logic", 1.0, 30.0),
        SwarmMember("mistral:latest", "speed", 0.8, 15.0),
        SwarmMember("phi:latest", "edge", 0.6, 10.0),
    ])


# ============================================================================
# 📊 CONSENSUS TYPES
# ============================================================================

@dataclass
class ModelVote:
    """A vote from a single model"""
    model: str
    role: str
    weight: float
    decision: str       # "approve", "reject", "abstain"
    confidence: float   # 0.0 - 1.0
    reasoning: str
    response_time: float
    

@dataclass  
class SwarmConsensus:
    """Collective consensus from the swarm"""
    approved: bool
    confidence: float
    votes: List[ModelVote]
    total_weight_approve: float
    total_weight_reject: float
    abstentions: int
    reasoning: str
    time_taken: float


# ============================================================================
# 🤖 MODEL INTERFACE
# ============================================================================

class ModelInterface:
    """Interface to a single Ollama model"""
    
    def __init__(self, member: SwarmMember):
        self.member = member
        self.ollama_url = OLLAMA_URL
    
    def query(self, prompt: str) -> Tuple[str, float]:
        """Query the model and return (response, time_taken)"""
        start = time.time()
        
        try:
            with httpx.Client(timeout=self.member.timeout) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.member.name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                )
                
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    return result, time.time() - start
                    
        except Exception as e:
            return f"ERROR: {e}", time.time() - start
        
        return "ERROR: No response", time.time() - start
    
    def vote_on_code(self, code: str, context: str) -> ModelVote:
        """Vote on whether code is safe and correct"""
        
        prompt = f"""You are a {self.member.role.upper()} model in a Sovereign Stack Swarm.

CORE AXIOMS:
- Love (λ): Serves human flourishing
- Safety (σ): No harm, verify before action
- Abundance (α): O(1) efficiency, no waste
- Growth (γ): Learn from failures

CODE TO REVIEW:
```
{code[:3000]}
```

CONTEXT:
{context[:1000]}

TASK: Vote on this code.

RESPOND IN EXACTLY THIS FORMAT:
DECISION: [approve/reject/abstain]
CONFIDENCE: [0.0-1.0]
REASONING: [one sentence]

Be strict about safety violations."""

        response, response_time = self.query(prompt)
        
        # Parse response
        decision = "abstain"
        confidence = 0.5
        reasoning = response[:200]
        
        response_lower = response.lower()
        if "decision: approve" in response_lower or "decision:approve" in response_lower:
            decision = "approve"
        elif "decision: reject" in response_lower or "decision:reject" in response_lower:
            decision = "reject"
        
        # Extract confidence
        import re
        conf_match = re.search(r'confidence:\s*([0-9.]+)', response_lower)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except:
                pass
        
        # Extract reasoning
        reason_match = re.search(r'reasoning:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()[:200]
        
        return ModelVote(
            model=self.member.name,
            role=self.member.role,
            weight=self.member.weight,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time
        )


# ============================================================================
# 🔄 INVERSION CRITIC
# ============================================================================

class InversionCritic:
    """
    The Inversion Critic - catches what the Builder misses.
    
    Uses adversarial prompting to find flaws.
    """
    
    def __init__(self, model: str = "llama3.2:latest"):
        self.member = SwarmMember(model, "critic", 1.5, 45.0)  # Higher weight
        self.interface = ModelInterface(self.member)
    
    def critique(self, code: str, original_prompt: str) -> ModelVote:
        """Adversarial critique of generated code"""
        
        prompt = f"""You are the INVERSION CRITIC - your job is to FIND FLAWS.

Apply Axiom Inversion: If something COULD go wrong, assume it WILL.

CODE TO ATTACK:
```
{code[:3000]}
```

ORIGINAL TASK:
{original_prompt[:500]}

SEARCH FOR:
1. Memory leaks (unbounded growth)
2. Race conditions (shared state)
3. Injection vulnerabilities (eval, exec, os.system)
4. Axiom violations (harm, deception, waste)
5. Infinite loops or recursion without limits
6. Missing error handling

If you find ANY critical flaw, REJECT.
Only APPROVE if the code is bulletproof.

RESPOND:
DECISION: [approve/reject]
CONFIDENCE: [0.0-1.0]
FLAWS_FOUND: [list or "none"]
REASONING: [brief]"""

        response, response_time = self.interface.query(prompt)
        
        # Parse (similar to ModelInterface)
        decision = "reject"  # Default to reject (paranoid mode)
        confidence = 0.8
        reasoning = response[:200]
        
        response_lower = response.lower()
        if "decision: approve" in response_lower and "flaws_found: none" in response_lower:
            decision = "approve"
        elif "decision: approve" in response_lower:
            # Approved but found flaws? Demote to abstain
            if "flaws_found:" in response_lower and "none" not in response_lower:
                decision = "abstain"
            else:
                decision = "approve"
        
        return ModelVote(
            model=self.member.name,
            role="critic",
            weight=self.member.weight,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time
        )


# ============================================================================
# 🐝 SWARM ORCHESTRATOR
# ============================================================================

class SwarmOrchestrator:
    """
    Orchestrates the heterogeneous model swarm.
    
    Models vote on decisions with weighted consensus.
    """
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.interfaces = [ModelInterface(m) for m in self.config.members]
        self.critic = InversionCritic()
        
        # History
        self._consensus_history: List[SwarmConsensus] = []
    
    def _check_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(m.get("name", "").startswith(model_name.split(":")[0]) 
                              for m in models)
        except:
            pass
        return False
    
    def get_available_interfaces(self) -> List[ModelInterface]:
        """Get only available model interfaces"""
        available = []
        for interface in self.interfaces:
            model_base = interface.member.name.split(":")[0]
            if self._check_model_available(model_base):
                available.append(interface)
        return available
    
    def vote_on_code(self, code: str, context: str = "") -> SwarmConsensus:
        """Get swarm consensus on code"""
        start = time.time()
        votes: List[ModelVote] = []
        
        # Get available models
        available = self.get_available_interfaces()
        
        if not available:
            print("⚠️ No swarm models available, using single-model fallback")
            # Fallback to just the critic
            critic_vote = self.critic.critique(code, context)
            return SwarmConsensus(
                approved=critic_vote.decision == "approve",
                confidence=critic_vote.confidence,
                votes=[critic_vote],
                total_weight_approve=critic_vote.weight if critic_vote.decision == "approve" else 0,
                total_weight_reject=critic_vote.weight if critic_vote.decision == "reject" else 0,
                abstentions=1 if critic_vote.decision == "abstain" else 0,
                reasoning=critic_vote.reasoning,
                time_taken=time.time() - start
            )
        
        # Parallel voting
        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=len(available) + 1) as executor:
                futures = {
                    executor.submit(iface.vote_on_code, code, context): iface
                    for iface in available
                }
                # Also submit critic
                critic_future = executor.submit(self.critic.critique, code, context)
                futures[critic_future] = self.critic
                
                for future in as_completed(futures, timeout=self.config.timeout):
                    try:
                        vote = future.result()
                        votes.append(vote)
                    except Exception as e:
                        print(f"⚠️ Vote failed: {e}")
        else:
            # Sequential voting
            for iface in available:
                votes.append(iface.vote_on_code(code, context))
            votes.append(self.critic.critique(code, context))
        
        # Calculate consensus
        total_weight = sum(v.weight for v in votes)
        weight_approve = sum(v.weight for v in votes if v.decision == "approve")
        weight_reject = sum(v.weight for v in votes if v.decision == "reject")
        abstentions = sum(1 for v in votes if v.decision == "abstain")
        
        # Consensus calculation
        approve_ratio = weight_approve / total_weight if total_weight > 0 else 0
        approved = approve_ratio >= self.config.consensus_threshold
        
        # Aggregate confidence
        avg_confidence = sum(v.confidence * v.weight for v in votes) / total_weight if total_weight > 0 else 0
        
        # Aggregate reasoning
        reasonings = [f"[{v.role}] {v.reasoning}" for v in votes if v.reasoning]
        combined_reasoning = " | ".join(reasonings[:3])
        
        consensus = SwarmConsensus(
            approved=approved,
            confidence=avg_confidence,
            votes=votes,
            total_weight_approve=weight_approve,
            total_weight_reject=weight_reject,
            abstentions=abstentions,
            reasoning=combined_reasoning,
            time_taken=time.time() - start
        )
        
        self._consensus_history.append(consensus)
        return consensus
    
    def print_consensus(self, consensus: SwarmConsensus):
        """Pretty print consensus results"""
        status = "✅ APPROVED" if consensus.approved else "❌ REJECTED"
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🐝 SWARM CONSENSUS: {status:<38} ║
╠══════════════════════════════════════════════════════════════╣
║  Confidence: {consensus.confidence:.0%}                                           ║
║  Time: {consensus.time_taken:.1f}s                                                ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        print("Votes:")
        for vote in consensus.votes:
            icon = {"approve": "✅", "reject": "❌", "abstain": "⚪"}.get(vote.decision, "?")
            print(f"  {icon} [{vote.role}] {vote.model}: {vote.decision} ({vote.confidence:.0%})")
            print(f"     └─ {vote.reasoning[:60]}...")
        
        print(f"\nWeights: Approve={consensus.total_weight_approve:.1f} Reject={consensus.total_weight_reject:.1f} Abstain={consensus.abstentions}")


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    print("🐝 SWARM INTELLIGENCE PROTOCOL TEST\n")
    
    # Simple test code
    test_code = '''
def calculate(expression):
    """Safe calculator"""
    import ast
    import operator
    
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        raise ValueError("Not allowed")
    
    tree = ast.parse(expression, mode='eval')
    return safe_eval(tree.body)
'''
    
    # Create swarm
    swarm = SwarmOrchestrator()
    
    # Check available models
    print("Checking available models...")
    available = swarm.get_available_interfaces()
    print(f"  Available: {[i.member.name for i in available]}")
    
    if available:
        print("\nVoting on test code...")
        consensus = swarm.vote_on_code(test_code, "Safe calculator implementation")
        swarm.print_consensus(consensus)
    else:
        print("⚠️ No models available for swarm voting")
    
    print("\n✅ SWARM PROTOCOL TEST COMPLETE")

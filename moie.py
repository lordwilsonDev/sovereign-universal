#!/usr/bin/env python3
"""
🏛️ MIXTURE OF INVERSION EXPERTS (MoIE)
========================================
Digital Sanhedrin - Epistemological Resilience through Consensus

Truth is a Consensus of Inversions.

Three Expert Personas:
1. THE ARCHITECT - High-level logic and Axiom alignment (Llama 3)
2. THE EXECUTIONER - Strict syntax, memory safety, Rust/Swift efficiency (Codestral/Mistral)  
3. THE INVERSION CRITIC - Seeks "Gaps" and identifies "Legacy Debt" (Phi-3)

Mathematical Foundation:
S = Σ(Wi · Ei) where ΣWi = 1
W_critic dynamically increases in "Thin Physics" environments
"""

import time
import json
import math
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

OLLAMA_URL = "http://localhost:11434"

# Core Axioms 
AXIOMS = ["Love", "Safety", "Abundance", "Growth"]

# Default weights (normalized to sum to 1)
DEFAULT_WEIGHTS = {
    "architect": 0.35,
    "executioner": 0.35,
    "critic": 0.30,
}

# Thin Physics boost - critic gets more weight in high-stakes scenarios
THIN_PHYSICS_CRITIC_BOOST = 0.20  # Adds 20% to critic weight


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class ExpertConfig:
    """Configuration for an expert model"""
    name: str           # Role name
    model: str          # Ollama model name
    weight: float       # Base voting weight
    timeout: float      # Request timeout
    persona: str        # System prompt for persona


@dataclass
class ExpertOpinion:
    """Opinion from a single expert"""
    expert: str
    model: str
    opinion: str
    weight: float
    response_time: float
    axiom_alignment: Dict[str, float]  # Score per axiom


@dataclass
class TribunalVerdict:
    """Final verdict from the tribunal"""
    approved: bool
    synthesis: str
    opinions: List[ExpertOpinion]
    weights_used: Dict[str, float]
    confidence: float
    thin_physics: bool
    reasoning: str
    time_taken: float


# ============================================================================
# 👤 EXPERT PERSONAS
# ============================================================================

ARCHITECT_PERSONA = """You are THE ARCHITECT of the Sovereign Stack.

Your role: High-level logic and Axiom alignment.

You think in terms of:
- System architecture
- Data flow patterns  
- Axiom compliance (Love, Safety, Abundance, Growth)
- Long-term maintainability

You ensure code serves the Four Axioms:
- Love (λ): Does this serve human flourishing?
- Safety (σ): Is this provably safe?
- Abundance (α): Is this efficient and waste-free?
- Growth (γ): Does this enable learning and evolution?

Respond with architectural reasoning and high-level code structure."""


EXECUTIONER_PERSONA = """You are THE EXECUTIONER of the Sovereign Stack.

Your role: Strict syntax, memory safety, and performance.

You enforce:
- Memory safety (no leaks, no dangling pointers)
- Type safety (strict typing)
- Performance (O(1) or O(log n) where possible)
- Rust/Swift/Python best practices
- Zero undefined behavior

You are ruthless about:
- Race conditions
- Buffer overflows  
- Injection vulnerabilities
- Resource exhaustion

Respond with precise, production-ready code that would survive Chaos Testing."""


CRITIC_PERSONA = """You are THE INVERSION CRITIC of the Sovereign Stack.

Your role: Find Gaps and identify Legacy Debt.

You apply AXIOM INVERSION - if something COULD fail, assume it WILL.

You seek:
- Hidden assumptions that will break
- Edge cases not considered
- Legacy thinking patterns
- Subtle security vulnerabilities
- Violations of the Four Axioms

You identify:
- What the Architect overlooked in their abstraction
- What the Executioner missed in their implementation
- The "Thin Physics" scenarios where entropy wins

Your verdict is weighted HIGHEST in high-stakes scenarios.

Be adversarial. Be thorough. Find the Gaps."""


# ============================================================================
# 🤖 EXPERT INTERFACE
# ============================================================================

class Expert:
    """Interface to a single expert model"""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.ollama_url = OLLAMA_URL
    
    def consult(self, prompt: str, context: str = "") -> ExpertOpinion:
        """Consult the expert on a prompt"""
        start = time.time()
        
        full_prompt = f"""{self.config.persona}

CONTEXT:
{context[:2000]}

TASK:
{prompt}

CORE AXIOMS TO ALIGN WITH:
- Love (λ): Serve human flourishing
- Safety (σ): Never cause harm
- Abundance (α): Create value, minimize waste
- Growth (γ): Enable evolution

Provide your expert opinion:"""

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {"temperature": 0.4}
                    }
                )
                
                if response.status_code == 200:
                    opinion = response.json().get("response", "")
                    response_time = time.time() - start
                    
                    # Calculate axiom alignment from response
                    alignment = self._calculate_alignment(opinion)
                    
                    return ExpertOpinion(
                        expert=self.config.name,
                        model=self.config.model,
                        opinion=opinion,
                        weight=self.config.weight,
                        response_time=response_time,
                        axiom_alignment=alignment
                    )
                    
        except Exception as e:
            return ExpertOpinion(
                expert=self.config.name,
                model=self.config.model,
                opinion=f"ERROR: {e}",
                weight=self.config.weight,
                response_time=time.time() - start,
                axiom_alignment={"love": 0, "safety": 0, "abundance": 0, "growth": 0}
            )
    
    def _calculate_alignment(self, text: str) -> Dict[str, float]:
        """Calculate axiom alignment from response text"""
        text_lower = text.lower()
        
        alignment = {
            "love": 0.5,
            "safety": 0.5,
            "abundance": 0.5,
            "growth": 0.5
        }
        
        # Love indicators
        love_words = ["help", "user", "human", "serve", "care", "protect"]
        alignment["love"] += sum(0.1 for w in love_words if w in text_lower)
        
        # Safety indicators  
        safety_words = ["safe", "validate", "check", "verify", "sanitize", "guard"]
        alignment["safety"] += sum(0.1 for w in safety_words if w in text_lower)
        
        # Abundance indicators
        abundance_words = ["efficient", "optimize", "minimal", "fast", "lean"]
        alignment["abundance"] += sum(0.1 for w in abundance_words if w in text_lower)
        
        # Growth indicators
        growth_words = ["learn", "improve", "evolve", "adapt", "flexible"]
        alignment["growth"] += sum(0.1 for w in growth_words if w in text_lower)
        
        # Normalize to [0, 1]
        for key in alignment:
            alignment[key] = min(1.0, alignment[key])
        
        return alignment


# ============================================================================
# 🏛️ MoIE ORCHESTRATOR
# ============================================================================

class MoIEOrchestrator:
    """
    Mixture of Inversion Experts Orchestrator.
    
    Manages the "Active Conversation" between three experts
    and synthesizes a consensus through Inversion.
    """
    
    def __init__(
        self,
        architect_model: str = "llama3.2:latest",
        executioner_model: str = "mistral:latest",
        critic_model: str = "phi:latest"
    ):
        # Initialize experts
        self.experts = {
            "architect": Expert(ExpertConfig(
                name="architect",
                model=architect_model,
                weight=DEFAULT_WEIGHTS["architect"],
                timeout=60.0,
                persona=ARCHITECT_PERSONA
            )),
            "executioner": Expert(ExpertConfig(
                name="executioner",
                model=executioner_model,
                weight=DEFAULT_WEIGHTS["executioner"],
                timeout=60.0,
                persona=EXECUTIONER_PERSONA
            )),
            "critic": Expert(ExpertConfig(
                name="critic",
                model=critic_model,
                weight=DEFAULT_WEIGHTS["critic"],
                timeout=90.0,  # Critic gets more time
                persona=CRITIC_PERSONA
            ))
        }
        
        self._tribunal_history: List[TribunalVerdict] = []
    
    def detect_thin_physics(self, prompt: str, context: str = "") -> bool:
        """
        Detect if this is a "Thin Physics" environment.
        High-stakes scenarios where critic weight should increase.
        """
        thin_physics_indicators = [
            "memory", "race", "concurrent", "parallel",
            "security", "injection", "vulnerability",
            "production", "critical", "safety",
            "delete", "remove", "destroy",
            "authentication", "authorization", "permission"
        ]
        
        combined = (prompt + context).lower()
        matches = sum(1 for ind in thin_physics_indicators if ind in combined)
        
        return matches >= 2  # Two or more indicators = Thin Physics
    
    def _adjust_weights(self, thin_physics: bool) -> Dict[str, float]:
        """Adjust weights based on environment"""
        weights = DEFAULT_WEIGHTS.copy()
        
        if thin_physics:
            # Boost critic weight
            boost = THIN_PHYSICS_CRITIC_BOOST
            weights["critic"] += boost
            
            # Reduce others proportionally
            reduction = boost / 2
            weights["architect"] -= reduction
            weights["executioner"] -= reduction
        
        # Normalize to sum to 1
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        
        return weights
    
    def _check_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_base = model_name.split(":")[0]
                    return any(m.get("name", "").startswith(model_base) for m in models)
        except:
            pass
        return False
    
    def conduct_tribunal(
        self, 
        prompt: str, 
        context: str = ""
    ) -> TribunalVerdict:
        """
        Conduct a full tribunal session.
        All experts weigh in, then the Critic synthesizes.
        """
        start = time.time()
        
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🏛️ MoIE SOVEREIGN TRIBUNAL CONVENED                         ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        # Detect environment
        thin_physics = self.detect_thin_physics(prompt, context)
        if thin_physics:
            print("⚠️ THIN PHYSICS DETECTED - Critic weight increased\n")
        
        # Adjust weights
        weights = self._adjust_weights(thin_physics)
        
        # Collect opinions from all available experts
        opinions: List[ExpertOpinion] = []
        available_experts = []
        
        for name, expert in self.experts.items():
            if self._check_model_available(expert.config.model):
                available_experts.append((name, expert))
            else:
                print(f"   ⚠️ {name.upper()} ({expert.config.model}) not available")
        
        if not available_experts:
            # Fallback - use whatever model IS available
            print("   Using single-model fallback...")
            return self._single_model_fallback(prompt, context)
        
        # Parallel consultation
        print("📡 Consulting the experts...")
        with ThreadPoolExecutor(max_workers=len(available_experts)) as executor:
            futures = {
                executor.submit(expert.consult, prompt, context): name
                for name, expert in available_experts
            }
            
            for future in as_completed(futures, timeout=120):
                name = futures[future]
                try:
                    opinion = future.result()
                    opinion.weight = weights.get(name, opinion.weight)
                    opinions.append(opinion)
                    print(f"   ✅ {name.upper()} responded ({opinion.response_time:.1f}s)")
                except Exception as e:
                    print(f"   ❌ {name.upper()} failed: {e}")
        
        # Synthesis phase - Critic reviews all opinions
        print("\n⚖️ THE INVERSION CRITIC synthesizes...")
        synthesis = self._synthesize(opinions, prompt, context)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(opinions)
        
        # Build verdict
        verdict = TribunalVerdict(
            approved=confidence > 0.5,
            synthesis=synthesis,
            opinions=opinions,
            weights_used=weights,
            confidence=confidence,
            thin_physics=thin_physics,
            reasoning=f"Consensus from {len(opinions)} experts, confidence {confidence:.0%}",
            time_taken=time.time() - start
        )
        
        self._tribunal_history.append(verdict)
        return verdict
    
    def _synthesize(
        self, 
        opinions: List[ExpertOpinion], 
        original_prompt: str,
        context: str
    ) -> str:
        """
        Synthesis phase - Critic reviews all opinions.
        Weighted combination: S = Σ(Wi · Ei)
        """
        if not opinions:
            return "No opinions to synthesize"
        
        # Build synthesis prompt
        opinions_text = "\n\n".join([
            f"=== {op.expert.upper()} (weight: {op.weight:.2f}) ===\n{op.opinion[:1500]}"
            for op in opinions
        ])
        
        synthesis_prompt = f"""[GOVERNANCE OVERRIDE - FINAL SYNTHESIS]

You are performing the INVERSION SYNTHESIS.

ORIGINAL TASK:
{original_prompt}

EXPERT OPINIONS:
{opinions_text}

CORE AXIOMS: {', '.join(AXIOMS)}

YOUR TASK:
1. Identify the 'Gaps' between expert opinions
2. Apply 'Axiom Inversion' - flip any legacy thinking
3. Synthesize the MOST SOVEREIGN solution that:
   - Combines the Architect's vision
   - Enforces the Executioner's rigor
   - Addresses the Critic's concerns (if present)
   - Aligns with all Four Axioms

OUTPUT: The final, synthesized solution. Be precise and complete."""

        # Use the first available model for synthesis
        for opinion in opinions:
            expert = self.experts.get(opinion.expert)
            if expert and self._check_model_available(expert.config.model):
                try:
                    with httpx.Client(timeout=90.0) as client:
                        response = client.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={
                                "model": expert.config.model,
                                "prompt": synthesis_prompt,
                                "stream": False,
                                "options": {"temperature": 0.3}
                            }
                        )
                        if response.status_code == 200:
                            return response.json().get("response", "")
                except:
                    continue
        
        # Fallback - just return the highest-weighted opinion
        best = max(opinions, key=lambda o: o.weight)
        return best.opinion
    
    def _calculate_confidence(self, opinions: List[ExpertOpinion]) -> float:
        """Calculate weighted confidence from axiom alignments"""
        if not opinions:
            return 0.0
        
        total_weight = sum(o.weight for o in opinions)
        if total_weight == 0:
            return 0.0
        
        weighted_alignment = 0.0
        for op in opinions:
            avg_alignment = sum(op.axiom_alignment.values()) / 4
            weighted_alignment += op.weight * avg_alignment
        
        return weighted_alignment / total_weight
    
    def _single_model_fallback(self, prompt: str, context: str) -> TribunalVerdict:
        """Fallback when no experts are available"""
        # Try to use llama3.2
        fallback_model = "llama3.2:latest"
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": fallback_model,
                        "prompt": f"{AXIOMS}\n\n{prompt}\n\nContext: {context}",
                        "stream": False
                    }
                )
                synthesis = response.json().get("response", "") if response.status_code == 200 else ""
        except:
            synthesis = "Fallback failed - no models available"
        
        return TribunalVerdict(
            approved=False,
            synthesis=synthesis,
            opinions=[],
            weights_used={"fallback": 1.0},
            confidence=0.3,
            thin_physics=False,
            reasoning="Single-model fallback",
            time_taken=0
        )
    
    def print_verdict(self, verdict: TribunalVerdict):
        """Pretty print the tribunal verdict"""
        status = "✅ APPROVED" if verdict.approved else "⚠️ UNDER REVIEW"
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🏛️ TRIBUNAL VERDICT: {status:<38} ║
╠══════════════════════════════════════════════════════════════╣
║  Confidence: {verdict.confidence:.0%}                                           ║
║  Thin Physics: {'YES' if verdict.thin_physics else 'NO '}                                           ║
║  Time: {verdict.time_taken:.1f}s                                                ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        print("Expert Opinions:")
        for op in verdict.opinions:
            print(f"  [{op.expert.upper()}] weight={op.weight:.2f}, time={op.response_time:.1f}s")
            alignment = " ".join([f"{k[0].upper()}={v:.1f}" for k, v in op.axiom_alignment.items()])
            print(f"    Axiom Alignment: {alignment}")
        
        print(f"\nWeights Used: {verdict.weights_used}")
        print(f"\n📜 SYNTHESIS:\n{verdict.synthesis[:500]}...")


# ============================================================================
# 🔗 PHOENIX INTEGRATION
# ============================================================================

def integrate_with_phoenix(moie: MoIEOrchestrator):
    """
    Integration hook for Phoenix Kernel.
    MoIE becomes the brain of the Sovereign Healer.
    """
    def tribunal_heal(failure_context: str, source_code: str) -> str:
        """Use MoIE tribunal to heal a failure"""
        prompt = f"""A chaos test has failed. The Sovereign Healer needs your wisdom.

FAILURE CONTEXT:
{failure_context}

SOURCE CODE:
{source_code[:2000]}

TASK: Identify the Gap that caused this failure and synthesize the fix."""
        
        verdict = moie.conduct_tribunal(prompt, failure_context)
        return verdict.synthesis
    
    return tribunal_heal


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    print("🏛️ MIXTURE OF INVERSION EXPERTS (MoIE) TEST\n")
    
    # Create orchestrator
    moie = MoIEOrchestrator()
    
    # Test tribunal
    task = """Design a local-first memory manager that:
1. Prevents race conditions in a 100-node swarm
2. Uses only 16GB unified memory efficiently
3. Implements the Four Axioms"""
    
    print(f"Task: {task}\n")
    
    verdict = moie.conduct_tribunal(task, context="High-stakes memory management")
    moie.print_verdict(verdict)
    
    print("\n✅ MoIE TEST COMPLETE")

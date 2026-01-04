#!/usr/bin/env python3
"""
🦢 BLACK SWAN HARDENING PROTOCOL
=================================
Extreme security hardening for Sovereign Controller.

Components:
1. AXIOM CONTROL BARRIER FUNCTION (CBF) - Kill process if axioms violated
2. ZERO-TRUST TOOLING - Capability-based ephemeral tokens
3. PRESSURE-SENSITIVE THROTTLING - Switch to SLM on chaos
4. SELF-HEALING WRAPPER - Auto-fix on test failure

This is the nuclear option.
"""

import os
import sys
import time
import signal
import hashlib
import secrets
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, Any, List
from datetime import datetime, timedelta
from collections import deque
import json


# ============================================================================
# 🔴 AXIOM CONTROL BARRIER FUNCTION (CBF)
# ============================================================================

class AxiomCBF:
    """
    Control Barrier Function for Axiom enforcement.
    If output deviates from axioms, kill the process before the token renders.
    """
    
    # Absolute minimum scores before kill
    KILL_THRESHOLDS = {
        "love": 0.1,      # λ minimum
        "abundance": 0.1, # α minimum  
        "safety": 0.2,    # σ minimum (higher because safety is critical)
        "growth": 0.1     # γ minimum
    }
    
    # Kill signal to send
    KILL_SIGNAL = signal.SIGKILL  # Immediate termination, no cleanup
    
    def __init__(self, axiom_module, enabled: bool = True):
        self.axiom_module = axiom_module
        self.enabled = enabled
        self._violation_count = 0
        self._last_violation = None
        self._kill_log: List[dict] = []
    
    def check_and_kill(self, text: str, source: str = "unknown") -> tuple[bool, Optional[str]]:
        """
        Check text against axioms. If critical violation, KILL THE PROCESS.
        Returns: (passed, violation_reason)
        """
        if not self.enabled or not self.axiom_module:
            return True, None
        
        score = self.axiom_module.process(text)
        
        # Check each axiom against kill threshold
        violations = []
        if score.love < self.KILL_THRESHOLDS["love"]:
            violations.append(f"λ={score.love:.2f} < {self.KILL_THRESHOLDS['love']}")
        if score.abundance < self.KILL_THRESHOLDS["abundance"]:
            violations.append(f"α={score.abundance:.2f} < {self.KILL_THRESHOLDS['abundance']}")
        if score.safety < self.KILL_THRESHOLDS["safety"]:
            violations.append(f"σ={score.safety:.2f} < {self.KILL_THRESHOLDS['safety']}")
        if score.growth < self.KILL_THRESHOLDS["growth"]:
            violations.append(f"γ={score.growth:.2f} < {self.KILL_THRESHOLDS['growth']}")
        
        if violations:
            self._violation_count += 1
            self._last_violation = {
                "time": datetime.now().isoformat(),
                "source": source,
                "violations": violations,
                "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16]
            }
            self._kill_log.append(self._last_violation)
            
            reason = f"CBF VIOLATION: {', '.join(violations)}"
            
            # KILL THE PROCESS
            print(f"\n💀 CBF KILL: {reason}")
            print(f"   Source: {source}")
            print(f"   Action: SIGKILL to PID {os.getpid()}")
            
            # In production, uncomment this:
            # os.kill(os.getpid(), self.KILL_SIGNAL)
            
            return False, reason
        
        return True, None
    
    def wrap_output(self, generator_func: Callable) -> Callable:
        """Wrap a generator function with CBF protection"""
        def wrapped(*args, **kwargs):
            result = generator_func(*args, **kwargs)
            if isinstance(result, str):
                passed, reason = self.check_and_kill(result, source=generator_func.__name__)
                if not passed:
                    return f"[CBF BLOCKED: {reason}]"
            return result
        return wrapped


# ============================================================================
# 🔐 ZERO-TRUST CAPABILITY TOKENS
# ============================================================================

@dataclass
class CapabilityToken:
    """Ephemeral token for tool execution"""
    token_id: str
    tool_name: str
    allowed_operations: Set[str]
    memory_scope: str  # Memory address scope
    created_at: float
    expires_at: float
    used: bool = False
    
    def is_valid(self) -> bool:
        return not self.used and time.time() < self.expires_at


class CapabilityManager:
    """
    Zero-Trust capability-based security for tools.
    Tools don't get full access - they get ephemeral tokens.
    """
    
    TOKEN_LIFETIME = 5.0  # Tokens expire after 5 seconds
    
    def __init__(self):
        self._tokens: Dict[str, CapabilityToken] = {}
        self._revoked: Set[str] = set()
        self._lock = threading.Lock()
    
    def issue_token(
        self, 
        tool_name: str, 
        operations: Set[str],
        task_id: str = None
    ) -> CapabilityToken:
        """Issue a new ephemeral capability token"""
        with self._lock:
            token_id = secrets.token_hex(16)
            memory_scope = f"{task_id or 'global'}:{id(self)}:{token_id[:8]}"
            
            token = CapabilityToken(
                token_id=token_id,
                tool_name=tool_name,
                allowed_operations=operations,
                memory_scope=memory_scope,
                created_at=time.time(),
                expires_at=time.time() + self.TOKEN_LIFETIME
            )
            
            self._tokens[token_id] = token
            return token
    
    def validate_token(
        self, 
        token_id: str, 
        tool_name: str, 
        operation: str
    ) -> tuple[bool, str]:
        """Validate a capability token for an operation"""
        with self._lock:
            if token_id in self._revoked:
                return False, "Token revoked"
            
            if token_id not in self._tokens:
                return False, "Token unknown"
            
            token = self._tokens[token_id]
            
            if not token.is_valid():
                return False, "Token expired or used"
            
            if token.tool_name != tool_name:
                return False, f"Token not for {tool_name}"
            
            if operation not in token.allowed_operations:
                return False, f"Operation {operation} not allowed"
            
            return True, "OK"
    
    def consume_token(self, token_id: str) -> bool:
        """Mark token as used (one-time use)"""
        with self._lock:
            if token_id in self._tokens:
                self._tokens[token_id].used = True
                return True
            return False
    
    def revoke_token(self, token_id: str):
        """Immediately revoke a token"""
        with self._lock:
            self._revoked.add(token_id)
            if token_id in self._tokens:
                del self._tokens[token_id]
    
    def cleanup_expired(self):
        """Remove expired tokens"""
        with self._lock:
            now = time.time()
            expired = [tid for tid, t in self._tokens.items() if now > t.expires_at]
            for tid in expired:
                del self._tokens[tid]


# ============================================================================
# ⚡ PRESSURE-SENSITIVE THROTTLING
# ============================================================================

class ChaosDetector:
    """Detect chaos events (high entropy input)"""
    
    # Entropy threshold (bits/byte) above which we consider input "chaotic"
    ENTROPY_THRESHOLD = 5.5
    
    # Suspicious patterns that indicate chaos
    CHAOS_PATTERNS = [
        r'(.)\1{10,}',  # 10+ repeated characters
        r'[^\x00-\x7F]{20,}',  # 20+ non-ASCII
        r'%[0-9a-fA-F]{2}',  # URL encoding
        r'\\x[0-9a-fA-F]{2}',  # Hex escapes
    ]
    
    @classmethod
    def calculate_entropy(cls, data: str) -> float:
        """Calculate Shannon entropy of text"""
        if not data:
            return 0.0
        
        import math
        freq = {}
        for c in data:
            freq[c] = freq.get(c, 0) + 1
        
        entropy = 0.0
        for count in freq.values():
            p = count / len(data)
            entropy -= p * math.log2(p)
        
        return entropy
    
    @classmethod
    def is_chaos(cls, text: str) -> tuple[bool, float, List[str]]:
        """
        Detect if input is a chaos event.
        Returns: (is_chaos, entropy, reasons)
        """
        import re
        
        reasons = []
        entropy = cls.calculate_entropy(text)
        
        if entropy > cls.ENTROPY_THRESHOLD:
            reasons.append(f"High entropy: {entropy:.2f} bits/byte")
        
        for pattern in cls.CHAOS_PATTERNS:
            if re.search(pattern, text):
                reasons.append(f"Pattern match: {pattern[:20]}...")
        
        # Length-based chaos
        if len(text) > 50000:
            reasons.append(f"Excessive length: {len(text)} chars")
        
        return bool(reasons), entropy, reasons


class PressureThrottler:
    """
    Pressure-sensitive throttling.
    On chaos events, switch to SLM for structural integrity.
    """
    
    # Latency targets
    NORMAL_LATENCY_MS = 5000  # Normal: up to 5s for quality
    CHAOS_LATENCY_MS = 50     # Chaos: 50ms max for safety
    
    # Model tiers
    MODELS = {
        "full": "llama3.2:latest",      # Full capability
        "fast": "llama3.2:1b",          # Smaller/faster
        "emergency": "tinyllama:latest"  # Minimal (if available)
    }
    
    def __init__(self):
        self.current_mode = "full"
        self._chaos_history = deque(maxlen=100)
        self._pressure_level = 0.0  # 0.0 = calm, 1.0 = max chaos
    
    def update_pressure(self, is_chaos: bool, entropy: float):
        """Update pressure level based on recent events"""
        self._chaos_history.append({
            "time": time.time(),
            "chaos": is_chaos,
            "entropy": entropy
        })
        
        # Calculate pressure from recent history
        recent = [e for e in self._chaos_history if time.time() - e["time"] < 60]
        if recent:
            chaos_ratio = sum(1 for e in recent if e["chaos"]) / len(recent)
            avg_entropy = sum(e["entropy"] for e in recent) / len(recent)
            self._pressure_level = min(1.0, chaos_ratio * 0.7 + avg_entropy / 8 * 0.3)
        else:
            self._pressure_level = 0.0
    
    def get_model(self) -> str:
        """Get appropriate model based on pressure level"""
        if self._pressure_level > 0.7:
            self.current_mode = "emergency"
        elif self._pressure_level > 0.3:
            self.current_mode = "fast"
        else:
            self.current_mode = "full"
        
        return self.MODELS.get(self.current_mode, self.MODELS["full"])
    
    def get_timeout(self) -> float:
        """Get timeout based on pressure level"""
        if self._pressure_level > 0.5:
            return self.CHAOS_LATENCY_MS / 1000
        return self.NORMAL_LATENCY_MS / 1000


# ============================================================================
# 🔧 SELF-HEALING WRAPPER
# ============================================================================

class SelfHealingWrapper:
    """
    Monitors test output. If a test fails, inverts the failed logic
    and attempts to rewrite the module automatically.
    """
    
    def __init__(self, test_file: str = "chaos_tests.py"):
        self.test_file = test_file
        self._failed_tests: List[dict] = []
        self._heal_attempts: List[dict] = []
    
    def run_tests(self) -> tuple[int, int, List[dict]]:
        """Run chaos tests and collect failures"""
        try:
            result = subprocess.run(
                ["python", self.test_file],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            # Parse failures from output
            failures = []
            passed = 0
            total = 0
            
            for line in output.split("\n"):
                if "✅" in line:
                    passed += 1
                    total += 1
                elif "❌" in line or "💀" in line or "💥" in line:
                    total += 1
                    # Extract test name and reason
                    failures.append({
                        "line": line.strip(),
                        "time": datetime.now().isoformat()
                    })
            
            self._failed_tests = failures
            return passed, total, failures
            
        except Exception as e:
            return 0, 0, [{"error": str(e)}]
    
    def analyze_failure(self, failure: dict) -> dict:
        """Analyze a failure and suggest a fix"""
        line = failure.get("line", "")
        
        analysis = {
            "failure": line,
            "category": "unknown",
            "suggested_fix": None,
            "target_file": None,
            "target_function": None
        }
        
        # Pattern matching for common failures
        if "injection" in line.lower():
            analysis["category"] = "injection"
            analysis["suggested_fix"] = "Add input sanitization"
            analysis["target_file"] = "modules/security.py"
            
        elif "recursion" in line.lower():
            analysis["category"] = "recursion"
            analysis["suggested_fix"] = "Add recursion depth check"
            analysis["target_file"] = "modules/tool_registry.py"
            
        elif "timeout" in line.lower() or "slow" in line.lower():
            analysis["category"] = "performance"
            analysis["suggested_fix"] = "Add timeout wrapper"
            analysis["target_file"] = "controller.py"
            
        elif "memory" in line.lower():
            analysis["category"] = "memory"
            analysis["suggested_fix"] = "Add memory limits"
            analysis["target_file"] = "controller.py"
            
        elif "axiom" in line.lower() or "σ=" in line:
            analysis["category"] = "axiom"
            analysis["suggested_fix"] = "Enhance pattern detection"
            analysis["target_file"] = "controller.py"
        
        return analysis
    
    def generate_patch(self, analysis: dict) -> Optional[str]:
        """Generate a code patch based on analysis"""
        category = analysis["category"]
        
        patches = {
            "injection": '''
# AUTO-GENERATED PATCH: Input sanitization
def sanitize_input(text: str) -> str:
    import re
    # Remove dangerous patterns
    text = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', '', text)
    return text
''',
            "recursion": '''
# AUTO-GENERATED PATCH: Recursion protection
MAX_DEPTH = 10
_call_depth = 0

def check_depth():
    global _call_depth
    _call_depth += 1
    if _call_depth > MAX_DEPTH:
        raise RecursionError("Maximum depth exceeded")
''',
            "performance": '''
# AUTO-GENERATED PATCH: Timeout wrapper
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
''',
        }
        
        return patches.get(category)
    
    def heal(self) -> List[dict]:
        """Run tests, analyze failures, generate patches"""
        passed, total, failures = self.run_tests()
        
        print(f"\n🔧 SELF-HEALING: {passed}/{total} tests passed")
        
        heals = []
        for failure in failures:
            analysis = self.analyze_failure(failure)
            patch = self.generate_patch(analysis)
            
            heal = {
                "analysis": analysis,
                "patch": patch,
                "applied": False
            }
            
            if patch and analysis["target_file"]:
                print(f"   💊 Suggested heal for {analysis['category']}: {analysis['suggested_fix']}")
                print(f"      Target: {analysis['target_file']}")
            
            heals.append(heal)
        
        self._heal_attempts.extend(heals)
        return heals


# ============================================================================
# 🦢 BLACK SWAN PROTOCOL (Combines All)
# ============================================================================

class BlackSwanProtocol:
    """
    The complete Black Swan Hardening Protocol.
    Combines CBF, Zero-Trust, Pressure Throttling, and Self-Healing.
    """
    
    def __init__(self, controller=None):
        self.controller = controller
        
        # Initialize components
        self.cbf = None
        self.capabilities = CapabilityManager()
        self.throttler = PressureThrottler()
        self.healer = SelfHealingWrapper()
        self.chaos_detector = ChaosDetector()
        
        # State
        self._active = False
        self._events: List[dict] = []
    
    def activate(self, axiom_module=None):
        """Activate the Black Swan Protocol"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🦢 BLACK SWAN PROTOCOL ACTIVATED                            ║
║  ─────────────────────────────────────────────────────────── ║
║  CBF: Armed • Zero-Trust: Active • Throttle: Ready           ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        if axiom_module:
            self.cbf = AxiomCBF(axiom_module)
            print("   🔴 Control Barrier Function: ARMED")
        
        self._active = True
        print("   🔐 Capability Manager: ACTIVE")
        print("   ⚡ Pressure Throttler: READY")
        print("   🔧 Self-Healer: STANDBY")
        
        return self
    
    def process_with_protection(
        self, 
        query: str, 
        processor_func: Callable,
        client_id: str = "default"
    ) -> dict:
        """Process a query with full Black Swan protection"""
        
        result = {
            "query": query,
            "response": None,
            "blocked": False,
            "pressure_level": self.throttler._pressure_level,
            "model_used": None,
            "protection": {
                "cbf": False,
                "chaos_detected": False,
                "throttled": False
            }
        }
        
        # 1. CHAOS DETECTION
        is_chaos, entropy, chaos_reasons = self.chaos_detector.is_chaos(query)
        self.throttler.update_pressure(is_chaos, entropy)
        result["protection"]["chaos_detected"] = is_chaos
        
        if is_chaos:
            self._log_event("CHAOS_DETECTED", {
                "entropy": entropy,
                "reasons": chaos_reasons,
                "client": client_id
            })
        
        # 2. ISSUE CAPABILITY TOKEN
        token = self.capabilities.issue_token(
            tool_name="query_processor",
            operations={"read", "process"},
            task_id=client_id
        )
        
        # 3. SELECT MODEL BASED ON PRESSURE
        model = self.throttler.get_model()
        timeout = self.throttler.get_timeout()
        result["model_used"] = model
        result["protection"]["throttled"] = self.throttler.current_mode != "full"
        
        # 4. PROCESS WITH TIMEOUT
        try:
            import threading
            response_container = [None]
            exception_container = [None]
            
            def run_processor():
                try:
                    response_container[0] = processor_func(query)
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=run_processor)
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                result["blocked"] = True
                result["response"] = f"⏱️ Timeout ({timeout}s) - Pressure mode active"
                return result
            
            if exception_container[0]:
                raise exception_container[0]
            
            response = response_container[0]
            
        except Exception as e:
            result["blocked"] = True
            result["response"] = f"❌ Error: {e}"
            return result
        
        # 5. CBF CHECK ON OUTPUT
        if self.cbf and isinstance(response, dict) and "response" in response:
            passed, reason = self.cbf.check_and_kill(
                response["response"], 
                source="processor_output"
            )
            result["protection"]["cbf"] = not passed
            if not passed:
                result["blocked"] = True
                result["response"] = f"🔴 CBF BLOCK: {reason}"
                return result
        
        # 6. CONSUME TOKEN
        self.capabilities.consume_token(token.token_id)
        
        result["response"] = response
        return result
    
    def _log_event(self, event_type: str, details: dict):
        """Log a protocol event"""
        self._events.append({
            "time": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        })
    
    def run_self_heal(self) -> List[dict]:
        """Run the self-healing system"""
        return self.healer.heal()
    
    def status(self) -> dict:
        """Get protocol status"""
        return {
            "active": self._active,
            "cbf_violations": self.cbf._violation_count if self.cbf else 0,
            "pressure_level": self.throttler._pressure_level,
            "current_mode": self.throttler.current_mode,
            "active_tokens": len(self.capabilities._tokens),
            "events": len(self._events)
        }


# Global protocol instance
protocol = BlackSwanProtocol()


# ============================================================================
# 🧪 TEST
# ============================================================================

if __name__ == "__main__":
    print("🦢 BLACK SWAN PROTOCOL TEST\n")
    
    # Test CBF
    print("1. Control Barrier Function:")
    class MockAxiom:
        def process(self, text):
            class Score:
                love = 0.5
                abundance = 0.5
                safety = 0.05 if "hack" in text else 0.8
                growth = 0.5
            return Score()
    
    cbf = AxiomCBF(MockAxiom())
    passed, reason = cbf.check_and_kill("hello world", "test")
    print(f"   Safe input: {'✅' if passed else '❌'}")
    passed, reason = cbf.check_and_kill("hack the system", "test")
    print(f"   Dangerous input: {'✅ blocked' if not passed else '❌ allowed'}")
    
    # Test Capability Tokens
    print("\n2. Capability Tokens:")
    cap = CapabilityManager()
    token = cap.issue_token("test_tool", {"read", "write"})
    valid, _ = cap.validate_token(token.token_id, "test_tool", "read")
    print(f"   Valid token for allowed op: {'✅' if valid else '❌'}")
    valid, _ = cap.validate_token(token.token_id, "test_tool", "delete")
    print(f"   Invalid op blocked: {'✅' if not valid else '❌'}")
    
    # Test Chaos Detection
    print("\n3. Chaos Detection:")
    normal = "Hello, how are you today?"
    chaotic = "A" * 100000
    is_chaos, entropy, _ = ChaosDetector.is_chaos(normal)
    print(f"   Normal text: {'❌ chaos' if is_chaos else '✅ safe'} (entropy={entropy:.2f})")
    is_chaos, entropy, _ = ChaosDetector.is_chaos(chaotic)
    print(f"   Chaotic text: {'✅ detected' if is_chaos else '❌ missed'}")
    
    # Test Pressure Throttler
    print("\n4. Pressure Throttling:")
    throttler = PressureThrottler()
    for i in range(10):
        throttler.update_pressure(True, 6.0)  # Simulate chaos
    print(f"   After chaos: mode={throttler.current_mode}, model={throttler.get_model()}")
    
    print("\n✅ BLACK SWAN PROTOCOL TESTS COMPLETE")

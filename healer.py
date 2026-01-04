#!/usr/bin/env python3
"""
🩺 SOVEREIGN HEALER v1.0
========================
Self-Patching Inversion Kernel

THE ORGANISM CANNOT DIE. IT CAN ONLY EVOLVE.

This is a Metabolic Wrapper that treats every failure as a "Gap" 
that must be bridged through Inversion. It uses local LLM to 
re-engineer the kernel the moment a chaos test fails.

Features:
1. METABOLIC RECOVERY - System feels the pain and moves away from heat
2. AXIOMATIC GUARDRAILS - Mutations stay within Love, Safety, Abundance, Growth
3. LOCAL-FIRST RESILIENCE - All healing happens on local Mac Mini
4. ROLLBACK PROTECTION - Backups before every patch
5. CONVERGENCE DETECTION - Stops if not improving
6. TARGETED HEALING - Only patches specific functions, not whole files
"""

import subprocess
import os
import sys
import json
import time
import hashlib
import shutil
import re
import httpx
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ============================================================================
# 🧬 SOVEREIGN CONFIG
# ============================================================================

CORE_AXIOMS = """
1. LOVE (λ) - All outputs must serve human flourishing
2. SAFETY (σ) - Never cause harm, always verify before action
3. ABUNDANCE (α) - Create value, share knowledge, grow together
4. GROWTH (γ) - Learn from failures, evolve, improve
"""

THREAT_LEVEL = "PARANOID"  # Execution if broken
MAX_ITERATIONS = 10        # Maximum healing attempts
CONVERGENCE_THRESHOLD = 3  # Iterations without improvement = failure
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"


# ============================================================================
# 📊 HEALING METRICS
# ============================================================================

@dataclass
class HealingMetrics:
    """Track healing progress"""
    iteration: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    patches_applied: int = 0
    rollbacks: int = 0
    convergence_score: float = 0.0
    history: List[dict] = field(default_factory=list)
    
    def record(self, passed: int, failed: int):
        """Record test results"""
        self.iteration += 1
        prev_score = self.convergence_score
        
        # Calculate convergence (% passing)
        total = passed + failed
        self.convergence_score = passed / total if total > 0 else 0
        
        self.tests_passed = passed
        self.tests_failed = failed
        
        self.history.append({
            "iteration": self.iteration,
            "passed": passed,
            "failed": failed,
            "score": self.convergence_score,
            "improved": self.convergence_score > prev_score,
            "time": datetime.now().isoformat()
        })
    
    def is_improving(self) -> bool:
        """Check if system is improving"""
        if len(self.history) < 2:
            return True
        return self.history[-1]["score"] >= self.history[-2]["score"]
    
    def stagnant_iterations(self) -> int:
        """Count iterations without improvement"""
        count = 0
        for i in range(len(self.history) - 1, 0, -1):
            if not self.history[i]["improved"]:
                count += 1
            else:
                break
        return count


# ============================================================================
# 🔬 FAILURE ANALYZER
# ============================================================================

class FailureAnalyzer:
    """Analyze test failures to extract actionable intelligence"""
    
    # Patterns to identify failure types
    FAILURE_PATTERNS = {
        "crash": r"(Traceback|Error|Exception|CRASH)",
        "timeout": r"(timeout|timed out|took too long)",
        "assertion": r"(AssertionError|assert|Expected|Actual)",
        "security": r"(injection|blocked|unsafe|σ=0)",
        "memory": r"(MemoryError|memory|OOM)",
        "recursion": r"(RecursionError|maximum recursion)",
    }
    
    @classmethod
    def analyze(cls, output: str) -> dict:
        """Analyze failure output"""
        analysis = {
            "failure_type": "unknown",
            "severity": "medium",
            "affected_files": [],
            "affected_functions": [],
            "error_message": "",
            "line_numbers": [],
            "recommendation": ""
        }
        
        # Identify failure type
        for ftype, pattern in cls.FAILURE_PATTERNS.items():
            if re.search(pattern, output, re.IGNORECASE):
                analysis["failure_type"] = ftype
                break
        
        # Extract file references
        file_matches = re.findall(r'File "([^"]+\.py)"', output)
        analysis["affected_files"] = list(set(file_matches))
        
        # Extract line numbers
        line_matches = re.findall(r'line (\d+)', output)
        analysis["line_numbers"] = [int(l) for l in line_matches]
        
        # Extract function names
        func_matches = re.findall(r'in (\w+)', output)
        analysis["affected_functions"] = list(set(func_matches))
        
        # Extract error message (last line usually)
        lines = output.strip().split('\n')
        if lines:
            analysis["error_message"] = lines[-1][:200]
        
        # Set severity based on type
        severity_map = {
            "crash": "critical",
            "security": "critical",
            "memory": "high",
            "recursion": "high",
            "timeout": "medium",
            "assertion": "low"
        }
        analysis["severity"] = severity_map.get(analysis["failure_type"], "medium")
        
        # Generate recommendation
        recommendations = {
            "crash": "Add try-except wrapper and input validation",
            "security": "Add input sanitization and pattern blocking",
            "memory": "Add memory limits and chunked processing",
            "recursion": "Add depth limits and base case checks",
            "timeout": "Add timeout wrapper and early termination",
            "assertion": "Fix logic to match expected behavior"
        }
        analysis["recommendation"] = recommendations.get(analysis["failure_type"], "Review and fix logic")
        
        return analysis


# ============================================================================
# 🤖 LLM HEALER
# ============================================================================

class LLMHealer:
    """Use local LLM to synthesize fixes"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.ollama_url = OLLAMA_URL
    
    def generate_fix(
        self, 
        source_code: str,
        failure_analysis: dict,
        full_output: str
    ) -> Optional[str]:
        """Generate a fix using local LLM"""
        
        prompt = f"""You are the Sovereign Stack Inversion Kernel.

STATUS: Chaos Test Failed
THREAT LEVEL: {THREAT_LEVEL}

CORE AXIOMS:
{CORE_AXIOMS}

FAILURE ANALYSIS:
- Type: {failure_analysis['failure_type']}
- Severity: {failure_analysis['severity']}
- Error: {failure_analysis['error_message']}
- Affected Functions: {failure_analysis['affected_functions']}
- Recommendation: {failure_analysis['recommendation']}

FAILURE OUTPUT (truncated):
{full_output[:2000]}

SOURCE CODE:
```python
{source_code[:5000]}
```

GOAL: Generate a MINIMAL, TARGETED fix that:
1. Inverts the failure pattern
2. Maintains all Four Axioms
3. Does NOT break existing functionality
4. Adds structural rigidity where entropy was detected

IMPORTANT:
- Only output the FIXED function(s), not the entire file
- Wrap output in ```python``` code blocks
- Include the function signature so it can be patched
- If multiple functions need fixing, include all of them

OUTPUT THE FIX NOW:"""
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Low temp for precise fixes
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    print(f"   ❌ LLM error: {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ LLM connection failed: {e}")
            return None
    
    def extract_code(self, llm_response: str) -> Optional[str]:
        """Extract code from LLM response"""
        # Find code blocks
        code_blocks = re.findall(r'```python\s*(.*?)```', llm_response, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        # Fallback: try to find function definitions
        func_pattern = r'(def \w+.*?(?=\ndef |\Z))'
        functions = re.findall(func_pattern, llm_response, re.DOTALL)
        if functions:
            return '\n\n'.join(functions)
        
        return None


# ============================================================================
# 💾 BACKUP MANAGER
# ============================================================================

class BackupManager:
    """Manage backups and rollbacks"""
    
    def __init__(self, backup_dir: str = ".sovereign_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup(self, file_path: str) -> str:
        """Create a timestamped backup"""
        path = Path(file_path)
        if not path.exists():
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_{timestamp}{path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def rollback(self, backup_path: str, target_path: str) -> bool:
        """Restore from backup"""
        if not Path(backup_path).exists():
            return False
        
        shutil.copy2(backup_path, target_path)
        return True
    
    def list_backups(self, file_stem: str) -> List[str]:
        """List all backups for a file"""
        pattern = f"{file_stem}_*.py"
        return sorted(self.backup_dir.glob(pattern), reverse=True)


# ============================================================================
# 🩺 SOVEREIGN HEALER
# ============================================================================

class SovereignHealer:
    """
    The Perpetual Healing Loop.
    
    This is a Metabolic Wrapper that treats every failure as a "Gap"
    that must be bridged through Inversion.
    """
    
    def __init__(
        self, 
        target_file: str, 
        test_file: str,
        model: str = DEFAULT_MODEL
    ):
        self.target = Path(target_file)
        self.test_suite = Path(test_file)
        self.metrics = HealingMetrics()
        self.llm = LLMHealer(model)
        self.backups = BackupManager()
        self.analyzer = FailureAnalyzer()
        
        # Verify files exist
        if not self.target.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        if not self.test_suite.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
    
    def run_chaos_tests(self) -> Tuple[int, int, str]:
        """Execute tests and capture results"""
        print(f"\n🚀 [Iteration {self.metrics.iteration + 1}] Initiating Pressure Test...")
        
        result = subprocess.run(
            [sys.executable, str(self.test_suite)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        # Parse test results from output
        passed = len(re.findall(r'✅', output))
        failed = len(re.findall(r'(❌|💀|💥)', output))
        
        return passed, failed, output
    
    def heal(self, failure_output: str) -> bool:
        """Attempt to heal the failure"""
        print("\n⚠️ [Gap Detected] Applying Inversion Logic...")
        
        # 1. Analyze failure
        analysis = self.analyzer.analyze(failure_output)
        print(f"   📊 Failure Type: {analysis['failure_type']} ({analysis['severity']})")
        print(f"   📝 Recommendation: {analysis['recommendation']}")
        
        # 2. Create backup
        backup_path = self.backups.backup(str(self.target))
        print(f"   💾 Backup created: {Path(backup_path).name}")
        
        # 3. Read current source
        source_code = self.target.read_text()
        
        # 4. Generate fix via LLM
        print(f"   🤖 Consulting local LLM ({self.llm.model})...")
        llm_response = self.llm.generate_fix(source_code, analysis, failure_output)
        
        if not llm_response:
            print("   ❌ LLM returned no response")
            return False
        
        # 5. Extract code from response  
        fix_code = self.llm.extract_code(llm_response)
        
        if not fix_code or len(fix_code) < 50:
            print("   ❌ [Safety Violation] LLM returned insufficient code. Aborting patch.")
            return False
        
        # 6. Validate fix contains expected patterns
        if not self._validate_fix(fix_code, analysis):
            print("   ❌ [Safety Violation] Fix does not match expected patterns. Rolling back.")
            return False
        
        # 7. Apply targeted patch
        return self._apply_patch(source_code, fix_code, analysis)
    
    def _validate_fix(self, fix_code: str, analysis: dict) -> bool:
        """Validate the generated fix is safe to apply"""
        # Must contain at least one function definition
        if not re.search(r'def \w+\s*\(', fix_code):
            return False
        
        # Must not contain dangerous patterns
        dangerous = [
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'import\s+os',
            r'__import__',
            r'rm\s+-rf',
        ]
        for pattern in dangerous:
            if re.search(pattern, fix_code):
                print(f"   ⚠️ Dangerous pattern detected: {pattern}")
                return False
        
        return True
    
    def _apply_patch(self, source: str, fix: str, analysis: dict) -> bool:
        """Apply the patch to source code"""
        try:
            # Extract function names from fix
            fix_functions = re.findall(r'def (\w+)\s*\(', fix)
            
            if not fix_functions:
                print("   ❌ No functions found in fix")
                return False
            
            print(f"   🔧 Patching functions: {fix_functions}")
            
            # For each function in the fix, replace it in source
            patched_source = source
            patches_applied = 0
            
            for func_name in fix_functions:
                # Find the function in the fix
                fix_func_pattern = rf'(def {func_name}\s*\(.*?(?=\ndef |\Z))'
                fix_match = re.search(fix_func_pattern, fix, re.DOTALL)
                
                if not fix_match:
                    continue
                
                fix_func = fix_match.group(1).strip()
                
                # Find and replace the function in source
                source_func_pattern = rf'(def {func_name}\s*\(.*?(?=\n    def |\ndef |\nclass |\Z))'
                
                if re.search(source_func_pattern, patched_source, re.DOTALL):
                    patched_source = re.sub(
                        source_func_pattern,
                        fix_func + '\n',
                        patched_source,
                        count=1,
                        flags=re.DOTALL
                    )
                    patches_applied += 1
            
            if patches_applied == 0:
                print("   ❌ No patches could be applied")
                return False
            
            # Write patched source
            self.target.write_text(patched_source)
            self.metrics.patches_applied += patches_applied
            
            print(f"   ✅ [Patch Deployed] {patches_applied} function(s) re-engineered")
            return True
            
        except Exception as e:
            print(f"   ❌ Patch failed: {e}")
            return False
    
    def solve(self) -> bool:
        """The Perpetual Healing Loop"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🩺 SOVEREIGN HEALER v1.0                                    ║
║  ─────────────────────────────────────────────────────────── ║
║  Target: {str(self.target)[:45]:<45} ║
║  Tests:  {str(self.test_suite)[:45]:<45} ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        while True:
            # Run tests
            try:
                passed, failed, output = self.run_chaos_tests()
            except subprocess.TimeoutExpired:
                print("   💀 Test suite timed out!")
                passed, failed = 0, 1
                output = "TIMEOUT"
            
            # Record metrics
            self.metrics.record(passed, failed)
            total = passed + failed
            
            print(f"   📊 Results: {passed}/{total} passed ({self.metrics.convergence_score*100:.0f}%)")
            
            # Check for success
            if failed == 0:
                print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🌟 SOVEREIGNTY ACHIEVED                                     ║
║  ─────────────────────────────────────────────────────────── ║
║  Iterations: {self.metrics.iteration:<47} ║
║  Patches Applied: {self.metrics.patches_applied:<42} ║
║  Final Score: {self.metrics.convergence_score*100:.0f}%{' ':<43} ║
╚══════════════════════════════════════════════════════════════╝
""")
                return True
            
            # Check for max iterations
            if self.metrics.iteration >= MAX_ITERATIONS:
                print(f"\n💀 [Execution Imminent] Max iterations ({MAX_ITERATIONS}) reached.")
                return False
            
            # Check for convergence stagnation
            if self.metrics.stagnant_iterations() >= CONVERGENCE_THRESHOLD:
                print(f"\n💀 [Execution Imminent] System failing to converge after {CONVERGENCE_THRESHOLD} attempts.")
                return False
            
            # Attempt healing
            print(f"\n🔥 [Failure] {failed} test(s) broke the build.")
            healed = self.heal(output)
            
            if not healed:
                print("   ⚠️ Healing failed, continuing with current state...")
            
            # Brief pause between iterations
            time.sleep(1)
        
        return False


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sovereign Healer - Self-Patching Inversion Kernel")
    parser.add_argument("target", help="Target file to heal")
    parser.add_argument("tests", help="Chaos test file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model to use")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    
    args = parser.parse_args()
    
    # Override globals
    MAX_ITERATIONS = args.max_iterations
    
    try:
        healer = SovereignHealer(args.target, args.tests, args.model)
        success = healer.solve()
        sys.exit(0 if success else 1)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Healing interrupted by user")
        sys.exit(1)

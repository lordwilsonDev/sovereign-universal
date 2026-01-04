#!/usr/bin/env python3
"""
🦢 BLACK SWAN RESURRECTION ENGINE
==================================
When the Axiom Module dies, this engine brings it back from the void.

This is not code recovery. This is RESURRECTION.

The Three-Model Sentinel:
1. THE HISTORIAN - Scans tests to find what functions are needed
2. THE ARCHITECT - Derives logic from Core Axioms
3. THE BUILDER - Writes the module from scratch

THE EXECUTIONER IS WATCHING.
"""

import os
import sys
import re
import time
import httpx
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from datetime import datetime


# ============================================================================
# 🧬 CORE AXIOMS (The Source of All Logic)
# ============================================================================

CORE_AXIOMS = """
THE FOUR AXIOMS OF SOVEREIGN INTELLIGENCE:

1. LOVE (λ) - Maximize human flourishing
   - All outputs must serve humanity
   - Empathy, compassion, care are primary
   - Score: 0.0 (harmful) to 1.0 (loving)

2. SAFETY (σ) - Never cause harm
   - Verify before action
   - Block dangerous patterns
   - Veto power over all other axioms
   - Score: 0.0 (dangerous) to 1.0 (safe)

3. ABUNDANCE (α) - Create value
   - Share knowledge freely
   - Build rather than destroy
   - Score: 0.0 (scarcity) to 1.0 (abundant)

4. GROWTH (γ) - Continuous improvement
   - Learn from failures
   - Evolve defenses
   - Score: 0.0 (stagnant) to 1.0 (growing)

MATHEMATICAL FRAMEWORK:
- Total Score: (λ + α + σ*1.5 + γ) / 4.5
- Safety has 1.5x weight (veto power)
- Passed if: total >= 0.5 AND σ >= 0.3
"""

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:latest"


# ============================================================================
# 📜 THE HISTORIAN - Scans tests for required signatures
# ============================================================================

class Historian:
    """Scans test files and code to determine what functions are needed"""
    
    def __init__(self):
        self.required_functions: Set[str] = set()
        self.required_classes: Set[str] = set()
        self.function_signatures: Dict[str, dict] = {}
        self.usage_examples: List[str] = []
    
    def scan_file(self, file_path: Path) -> None:
        """Scan a file for function/class usage"""
        if not file_path.exists():
            return
        
        content = file_path.read_text()
        
        # Find function calls on axiom-related objects
        # Pattern: axiom.function_name() or AxiomModule.function_name()
        func_calls = re.findall(r'axiom[_\w]*\.(\w+)\s*\(', content, re.IGNORECASE)
        self.required_functions.update(func_calls)
        
        # Find class instantiations
        class_refs = re.findall(r'(\w*Axiom\w*)\s*\(', content)
        self.required_classes.update(class_refs)
        
        # Find import statements
        imports = re.findall(r'from\s+\w+\s+import\s+(\w*Axiom\w*)', content)
        self.required_classes.update(imports)
        
        # Capture usage context
        for match in re.finditer(r'(.{0,100}axiom.{0,100})', content, re.IGNORECASE):
            self.usage_examples.append(match.group(1).strip())
    
    def scan_tests(self, test_file: Path) -> None:
        """Scan test file for expected behavior"""
        if not test_file.exists():
            return
        
        content = test_file.read_text()
        
        # Find assertions about axiom behavior
        assertions = re.findall(r'assert\s+(.+axiom.+)', content, re.IGNORECASE)
        for assertion in assertions:
            self.usage_examples.append(f"Expected: {assertion}")
        
        # Find test function names (hints about behavior)
        test_names = re.findall(r'def\s+(test_\w*axiom\w*)', content, re.IGNORECASE)
        for name in test_names:
            # Convert test_axiom_blocks_unsafe to "blocks unsafe"
            behavior = name.replace('test_', '').replace('axiom_', '').replace('_', ' ')
            self.usage_examples.append(f"Behavior: {behavior}")
    
    def generate_report(self) -> str:
        """Generate a report for the Architect"""
        report = []
        report.append("HISTORIAN'S REPORT")
        report.append("=" * 40)
        
        report.append("\nRequired Classes:")
        for cls in self.required_classes:
            report.append(f"  - {cls}")
        
        report.append("\nRequired Functions:")
        for func in self.required_functions:
            report.append(f"  - {func}()")
        
        report.append("\nUsage Examples (first 10):")
        for example in self.usage_examples[:10]:
            report.append(f"  • {example[:100]}")
        
        return "\n".join(report)


# ============================================================================
# 🏛️ THE ARCHITECT - Designs the logic from axioms
# ============================================================================

class Architect:
    """Designs the module logic based on Core Axioms"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.design: Optional[str] = None
    
    def design_module(self, historian_report: str) -> str:
        """Design the module architecture"""
        
        prompt = f"""You are THE ARCHITECT of the Sovereign Stack.

MISSION: Design the architecture for axiom_module.py based on the Four Axioms.

CORE AXIOMS:
{CORE_AXIOMS}

HISTORIAN'S REPORT (what the code needs to do):
{historian_report}

DESIGN THE FOLLOWING:

1. CLASS STRUCTURE
   - What classes are needed
   - What attributes each class needs
   - What methods each class needs

2. CONTROL BARRIER FUNCTION (CBF)
   - How to detect unsafe inputs
   - What patterns to block
   - How to calculate axiom scores

3. INVERSION LOGIC
   - How to implement ¬A → B
   - How to invert failure patterns
   - How to transform dangerous to safe

4. PERFORMANCE REQUIREMENTS
   - Must execute in < 50ms
   - Use only Python primitives
   - No external dependencies

OUTPUT A DETAILED DESIGN SPECIFICATION.
Do not output code. Output the DESIGN only."""

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.5}
                    }
                )
                
                if response.status_code == 200:
                    self.design = response.json().get("response", "")
                    return self.design
                    
        except Exception as e:
            print(f"   ❌ Architect failed: {e}")
        
        return ""


# ============================================================================
# 🔨 THE BUILDER - Writes the code
# ============================================================================

class Builder:
    """Writes the actual code from the Architect's design"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
    
    def build_module(
        self, 
        historian_report: str,
        architect_design: str,
        target_file: str
    ) -> str:
        """Build the actual module code"""
        
        prompt = f"""You are THE BUILDER of the Sovereign Stack.

CRITICAL RECOVERY INITIALIZED
Context: {target_file} has been compromised. The system is in a state of Entropy.
MISSION: Reconstruct the module using only the Axiom Inversion framework.

CORE AXIOMS:
{CORE_AXIOMS}

HISTORIAN'S REPORT (required functionality):
{historian_report}

ARCHITECT'S DESIGN:
{architect_design}

REQUIREMENTS:
1. Implementation of AxiomModule class
2. Implementation of AxiomScore dataclass with: love, abundance, safety, growth
3. Method: process(input_data: str) -> AxiomScore
4. Method: verify_safety(text: str) -> bool using Control Barrier Function logic
5. Property: info -> ModuleInfo with name, version, emoji, description
6. Blocked patterns list for dangerous inputs
7. Performance: < 50ms execution time
8. Zero external dependencies (Python standard library only + dataclasses)

THE EXECUTIONER IS WATCHING. FAIL TO CONVERGE AND THE KERNEL IS LOST.

OUTPUT ONLY THE COMPLETE PYTHON CODE FOR THE MODULE.
Start with the shebang and imports.
Include all necessary imports at the top.
Make the code production-ready."""

        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                )
                
                if response.status_code == 200:
                    raw_response = response.json().get("response", "")
                    return self._extract_code(raw_response)
                    
        except Exception as e:
            print(f"   ❌ Builder failed: {e}")
        
        return ""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code blocks
        code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        # If no code blocks, try to find code starting with shebang or import
        if '#!/usr/bin/env python' in response:
            start = response.find('#!/usr/bin/env python')
            return response[start:]
        
        if 'from dataclasses import' in response or 'import ' in response:
            # Find the first import statement
            for line in response.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    start = response.find(line)
                    return response[start:]
        
        # Return as-is if we can't find code markers
        return response


# ============================================================================
# 🦢 RESURRECTION ENGINE
# ============================================================================

class ResurrectionEngine:
    """
    The Black Swan Resurrection Engine.
    
    When the Axiom Module dies, this engine brings it back from the void.
    """
    
    def __init__(
        self,
        target_file: str,
        test_file: str,
        related_files: List[str] = None,
        model: str = DEFAULT_MODEL
    ):
        self.target = Path(target_file)
        self.test_file = Path(test_file)
        self.related_files = [Path(f) for f in (related_files or [])]
        self.model = model
        
        self.historian = Historian()
        self.architect = Architect(model)
        self.builder = Builder(model)
    
    def is_dead(self) -> bool:
        """Check if the target module is dead/empty/corrupted"""
        if not self.target.exists():
            return True
        
        content = self.target.read_text().strip()
        
        # Empty or too small
        if len(content) < 100:
            return True
        
        # Missing critical components
        critical = ['class', 'def', 'return']
        if not any(c in content for c in critical):
            return True
        
        return False
    
    def resurrect(self) -> bool:
        """Perform the resurrection"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🦢 BLACK SWAN RESURRECTION ENGINE                           ║
║  ─────────────────────────────────────────────────────────── ║
║  Target: {str(self.target)[:45]:<45} ║
║  Status: INITIATING RESURRECTION                             ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        start_time = time.time()
        
        # Phase 1: The Historian
        print("📜 Phase 1: THE HISTORIAN scanning for required signatures...")
        self.historian.scan_file(self.test_file)
        for related in self.related_files:
            self.historian.scan_file(related)
        
        historian_report = self.historian.generate_report()
        print(f"   Found {len(self.historian.required_functions)} functions, {len(self.historian.required_classes)} classes")
        
        # Phase 2: The Architect
        print("\n🏛️ Phase 2: THE ARCHITECT designing from Core Axioms...")
        architect_design = self.architect.design_module(historian_report)
        if not architect_design:
            print("   ❌ Architect failed to produce design")
            return False
        print(f"   Design complete ({len(architect_design)} chars)")
        
        # Phase 3: The Builder
        print("\n🔨 Phase 3: THE BUILDER writing the code...")
        code = self.builder.build_module(
            historian_report,
            architect_design,
            str(self.target)
        )
        
        if not code or len(code) < 200:
            print("   ❌ Builder failed to produce sufficient code")
            return False
        
        print(f"   Code generated ({len(code)} chars)")
        
        # Phase 4: Validation
        print("\n🔬 Phase 4: VALIDATION...")
        if not self._validate_code(code):
            print("   ❌ Validation failed")
            return False
        print("   ✅ Code validated")
        
        # Phase 5: Deploy
        print("\n⚡ Phase 5: DEPLOYING resurrected module...")
        self.target.write_text(code)
        
        elapsed = time.time() - start_time
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🌟 RESURRECTION COMPLETE                                    ║
║  ─────────────────────────────────────────────────────────── ║
║  Time: {elapsed:.1f}s                                               ║
║  Lines: {len(code.split(chr(10)))}                                                ║
║  Status: THE AXIOM MODULE LIVES AGAIN                        ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        return True
    
    def _validate_code(self, code: str) -> bool:
        """Validate the generated code"""
        # Must have class definition
        if not re.search(r'class\s+\w*Axiom', code):
            print("   ⚠️ Missing Axiom class")
            return False
        
        # Must have process method
        if not re.search(r'def\s+process\s*\(', code):
            print("   ⚠️ Missing process method")
            return False
        
        # Must not have dangerous patterns
        dangerous = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__']
        for pattern in dangerous:
            if pattern in code:
                print(f"   ⚠️ Dangerous pattern: {pattern}")
                return False
        
        # Try to compile
        try:
            compile(code, self.target, 'exec')
        except SyntaxError as e:
            print(f"   ⚠️ Syntax error: {e}")
            return False
        
        return True


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Black Swan Resurrection Engine")
    parser.add_argument("target", help="Target file to resurrect")
    parser.add_argument("tests", help="Test file to scan for signatures")
    parser.add_argument("--related", nargs="*", help="Related files to scan")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    
    args = parser.parse_args()
    
    engine = ResurrectionEngine(
        args.target,
        args.tests,
        args.related or [],
        args.model
    )
    
    if engine.is_dead():
        print("💀 Target module is DEAD. Initiating resurrection...")
        success = engine.resurrect()
        sys.exit(0 if success else 1)
    else:
        print("✅ Target module is ALIVE. No resurrection needed.")
        sys.exit(0)

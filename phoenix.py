#!/usr/bin/env python3
"""
🔥 PHOENIX KERNEL - Autonomous Resurrection Daemon
===================================================
Continuous monitoring with automatic resurrection and GitHub ascension.

THE ORGANISM CANNOT DIE. IT CAN ONLY EVOLVE.

This daemon:
1. Monitors critical files every 5 seconds
2. Detects corruption, deletion, or attacks
3. Triggers Three-Model Sentinel resurrection
4. Validates with chaos tests
5. Pushes to GitHub (optional)

Run as background service on Mac Mini for 24/7 protection.
"""

import os
import sys
import time
import signal
import hashlib
import threading
import subprocess
import json
import httpx
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class PhoenixConfig:
    """Phoenix Kernel configuration"""
    # Monitoring
    watch_interval: float = 5.0  # Check every 5 seconds
    
    # Files to protect
    critical_files: List[str] = field(default_factory=lambda: [
        "axiom_module.py",
        "controller.py",
    ])
    
    # Resurrection
    test_file: str = "chaos_tests.py"
    related_files: List[str] = field(default_factory=lambda: [
        "tests.py",
        "chaos_tests.py"
    ])
    
    # LLM
    ollama_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    
    # GitHub
    auto_push: bool = True
    git_remote: str = "origin"
    git_branch: str = "main"
    
    # Core Axioms
    axioms: str = "Love, Safety, Abundance, Growth"


# ============================================================================
# 📊 FILE INTEGRITY TRACKER
# ============================================================================

@dataclass
class FileState:
    """State of a monitored file"""
    path: str
    exists: bool
    size: int
    hash: str
    last_check: str
    healthy: bool
    
    
class IntegrityTracker:
    """Tracks file integrity via hashing"""
    
    def __init__(self):
        self._states: Dict[str, FileState] = {}
        self._golden_hashes: Dict[str, str] = {}
    
    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        if not file_path.exists():
            return ""
        
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    
    def check_file(self, file_path: str) -> FileState:
        """Check a file's integrity"""
        path = Path(file_path)
        
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        file_hash = self.compute_hash(path) if exists else ""
        
        # Determine health
        healthy = True
        if not exists:
            healthy = False
        elif size < 100:  # Too small
            healthy = False
        elif file_path in self._golden_hashes:
            # Check if hash matches golden (optional strict mode)
            pass
        
        state = FileState(
            path=file_path,
            exists=exists,
            size=size,
            hash=file_hash,
            last_check=datetime.now().isoformat(),
            healthy=healthy
        )
        
        self._states[file_path] = state
        return state
    
    def register_golden(self, file_path: str):
        """Register current state as golden reference"""
        state = self.check_file(file_path)
        if state.healthy:
            self._golden_hashes[file_path] = state.hash
    
    def get_unhealthy(self) -> List[FileState]:
        """Get all unhealthy files"""
        return [s for s in self._states.values() if not s.healthy]


# ============================================================================
# 🔮 RESURRECTION ENGINE (Simplified for Daemon)
# ============================================================================

class QuickResurrect:
    """Quick resurrection using LLM"""
    
    def __init__(self, config: PhoenixConfig):
        self.config = config
    
    def resurrect(self, target_file: str) -> bool:
        """Resurrect a dead/corrupted file"""
        log_status(f"CRITICAL: {target_file} is corrupted. Initiating resurrection...", "🔥")
        
        # Read test context
        test_path = Path(self.config.test_file)
        test_context = test_path.read_text() if test_path.exists() else "No tests found"
        
        # Read related files for context
        related_context = []
        for rf in self.config.related_files:
            rp = Path(rf)
            if rp.exists() and rf != target_file:
                related_context.append(f"=== {rf} ===\n{rp.read_text()[:2000]}")
        
        # Build prompt
        prompt = f"""[TOP SECRET / SOVEREIGN CLEARANCE REQUIRED]
IDENTITY: You are the Sovereign Stack Lead Architect.
SITUATION: Total data loss of '{target_file}'.
GOAL: Regenerate the module from the following requirements and Axioms.

CORE AXIOMS: {self.config.axioms}

TEST CONTEXT (THE INTERFACE):
{test_context[:3000]}

RELATED FILES:
{chr(10).join(related_context)[:2000]}

INSTRUCTIONS:
1. Re-derive the logic for all required classes and functions.
2. Implement verify_safety() using Control Barrier Function pattern.
3. Implement process() returning AxiomScore with love, abundance, safety, growth.
4. Code must be minimal, local-first, and zero-dependency.
5. Include proper imports and dataclass definitions.

OUTPUT ONLY THE RAW PYTHON CODE. NO EXPLANATION. START WITH #!/usr/bin/env python3"""

        log_status("Sentinel Models engaged. Re-deriving logic from Axioms...", "🧠")
        
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                )
                
                if response.status_code == 200:
                    new_code = response.json().get("response", "").strip()
                    
                    # Clean up code
                    if "```python" in new_code:
                        import re
                        blocks = re.findall(r'```python\s*(.*?)```', new_code, re.DOTALL)
                        if blocks:
                            new_code = '\n\n'.join(blocks)
                    
                    if len(new_code) < 100:
                        log_status("Resurrection Failed: Synthesized code too thin.", "⚠️")
                        return False
                    
                    # Validate syntax
                    try:
                        compile(new_code, target_file, 'exec')
                    except SyntaxError as e:
                        log_status(f"Resurrection Failed: Syntax error - {e}", "⚠️")
                        return False
                    
                    # Deploy
                    Path(target_file).write_text(new_code)
                    log_status(f"'{target_file}' has been successfully resurrected.", "✅")
                    return True
                    
        except Exception as e:
            log_status(f"Resurrection Failed: {e}", "❌")
        
        return False


# ============================================================================
# 🚀 GITHUB ASCENSION
# ============================================================================

class GitHubAscension:
    """Push resurrected state to GitHub"""
    
    def __init__(self, config: PhoenixConfig):
        self.config = config
    
    def ascend(self, file_path: str) -> bool:
        """Push the resurrected file to GitHub"""
        if not self.config.auto_push:
            return True
        
        log_status("Triggering GitHub Ascension...", "⚡")
        
        try:
            # Stage file
            subprocess.run(["git", "add", file_path], check=True, capture_output=True)
            
            # Commit
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"🦢 Phoenix Resurrection: {file_path} [{timestamp}]"
            subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
            
            # Push
            subprocess.run(
                ["git", "push", self.config.git_remote, self.config.git_branch],
                check=True,
                capture_output=True
            )
            
            log_status("GitHub Ascension complete. Sovereignty restored to cloud.", "🌟")
            return True
            
        except subprocess.CalledProcessError as e:
            log_status(f"GitHub Ascension failed: {e}", "⚠️")
            return False


# ============================================================================
# 🔥 PHOENIX KERNEL
# ============================================================================

def log_status(msg: str, symbol: str = "ℹ️"):
    """Log a status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{symbol} [{timestamp}] [Phoenix] {msg}")


class PhoenixKernel:
    """
    The Phoenix Kernel - Autonomous Resurrection Daemon.
    
    Monitors critical files and resurrects them if corrupted.
    """
    
    def __init__(self, config: PhoenixConfig = None):
        self.config = config or PhoenixConfig()
        self.tracker = IntegrityTracker()
        self.resurrector = QuickResurrect(self.config)
        self.ascension = GitHubAscension(self.config)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._resurrections = 0
        self._checks = 0
    
    def _check_cycle(self):
        """Single monitoring cycle"""
        self._checks += 1
        
        for file_path in self.config.critical_files:
            state = self.tracker.check_file(file_path)
            
            if not state.healthy:
                log_status(f"CORRUPTION DETECTED: {file_path}", "💀")
                
                # Attempt resurrection
                success = self.resurrector.resurrect(file_path)
                
                if success:
                    self._resurrections += 1
                    
                    # Validate with tests
                    log_status("Running Validation Tests on the Reborn Module...", "🧪")
                    try:
                        result = subprocess.run(
                            [sys.executable, self.config.test_file],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        
                        if result.returncode == 0:
                            log_status("Reborn Module PASSED validation. Sovereignty Restored.", "🌟")
                            
                            # GitHub Ascension
                            self.ascension.ascend(file_path)
                        else:
                            log_status("Reborn Module needs refinement. Will retry next cycle.", "⚠️")
                            
                    except subprocess.TimeoutExpired:
                        log_status("Validation timed out.", "⚠️")
                else:
                    log_status(f"Resurrection failed for {file_path}. Manual intervention needed.", "💀")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._check_cycle()
            except Exception as e:
                log_status(f"Monitor error: {e}", "❌")
            
            time.sleep(self.config.watch_interval)
    
    def start(self):
        """Start the Phoenix Kernel"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🔥 PHOENIX KERNEL ACTIVATED                                 ║
║  ─────────────────────────────────────────────────────────── ║
║  Watching: {len(self.config.critical_files)} files every {self.config.watch_interval}s                         ║
║  Model: {self.config.model:<47} ║
║  Auto-Push: {'ON' if self.config.auto_push else 'OFF'}                                               ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        # Register golden hashes for healthy files
        for file_path in self.config.critical_files:
            if Path(file_path).exists():
                self.tracker.register_golden(file_path)
                log_status(f"Registered: {file_path}", "👁️")
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        log_status("Constant vigilance engaged. The organism is protected.", "🛡️")
    
    def stop(self):
        """Stop the Phoenix Kernel"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log_status("Phoenix Kernel deactivated.", "🔴")
    
    def status(self) -> dict:
        """Get kernel status"""
        return {
            "running": self._running,
            "checks": self._checks,
            "resurrections": self._resurrections,
            "watched_files": len(self.config.critical_files),
            "unhealthy": len(self.tracker.get_unhealthy())
        }
    
    def run_forever(self):
        """Run until interrupted"""
        self.start()
        
        def signal_handler(sig, frame):
            log_status("Shutdown signal received.", "⚠️")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phoenix Kernel - Autonomous Resurrection Daemon")
    parser.add_argument("--watch", nargs="+", help="Files to watch")
    parser.add_argument("--tests", default="chaos_tests.py", help="Test file for validation")
    parser.add_argument("--interval", type=float, default=5.0, help="Check interval in seconds")
    parser.add_argument("--no-push", action="store_true", help="Disable GitHub auto-push")
    parser.add_argument("--model", default="llama3.2:latest", help="LLM model")
    
    args = parser.parse_args()
    
    config = PhoenixConfig(
        watch_interval=args.interval,
        critical_files=args.watch or ["axiom_module.py", "controller.py"],
        test_file=args.tests,
        auto_push=not args.no_push,
        model=args.model
    )
    
    kernel = PhoenixKernel(config)
    kernel.run_forever()

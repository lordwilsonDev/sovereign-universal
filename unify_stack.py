#!/usr/bin/env python3
"""
🌟 SOVEREIGN UNIFICATION WRAPPER
==================================
The Observer Effect: "By observing the system, you collapse the wave function."

Launches and orchestrates the entire Sovereign Stack:
- Telemetry Sniffer (The Senses)
- Phoenix Kernel (The Resurrection Engine)
- Sovereign Healer (The Immune System)
- MoIE Orchestrator (The Brain)
- SwiftUI Dashboard (The Face)

Zero-Entropy Sovereignty through Local IPC.
"""

import subprocess
import threading
import signal
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from queue import Queue


# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

@dataclass
class ModuleConfig:
    """Configuration for a Sovereign module"""
    name: str
    command: str
    enabled: bool = True
    restart_on_crash: bool = True
    startup_delay: float = 0.5
    health_check: Optional[Callable] = None


# Module paths (relative to sovereign_universal)
BASE_DIR = Path(__file__).parent

CONFIG = {
    "telemetry": ModuleConfig(
        name="Telemetry Sniffer",
        command=f"python3 {BASE_DIR}/telemetry.py",
        enabled=True,
        restart_on_crash=True
    ),
    "phoenix": ModuleConfig(
        name="Phoenix Kernel",
        command=f"python3 {BASE_DIR}/phoenix.py --watch axiom_module.py controller.py --interval 10",
        enabled=True,
        restart_on_crash=True
    ),
    "healer": ModuleConfig(
        name="Sovereign Healer",
        command=f"python3 {BASE_DIR}/healer.py {BASE_DIR}/controller.py {BASE_DIR}/chaos_tests.py --max-iterations 1",
        enabled=False,  # Only run on-demand
        restart_on_crash=False
    ),
    "moie": ModuleConfig(
        name="MoIE Brain",
        command=f"python3 -c \"from moie import MoIEOrchestrator; print('MoIE Ready')\"",
        enabled=True,
        restart_on_crash=True
    ),
    "dashboard": ModuleConfig(
        name="Sovereign Dashboard",
        command="open -a SovereignDashboard",  # macOS app
        enabled=False,  # Requires compiled Swift app
        restart_on_crash=False
    )
}


# ============================================================================
# 📊 PROCESS MANAGER
# ============================================================================

@dataclass
class ProcessState:
    """State of a managed process"""
    config: ModuleConfig
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_output: str = ""
    status: str = "stopped"  # stopped, starting, running, crashed


class ProcessManager:
    """Manages a single subprocess"""
    
    def __init__(self, config: ModuleConfig, event_queue: Queue):
        self.config = config
        self.state = ProcessState(config=config)
        self.event_queue = event_queue
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        """Start the process"""
        if self.state.process and self.state.process.poll() is None:
            return True  # Already running
        
        self.state.status = "starting"
        self._emit_event("starting", f"Starting {self.config.name}...")
        
        try:
            self.state.process = subprocess.Popen(
                self.config.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR)
            )
            self.state.start_time = datetime.now()
            self.state.status = "running"
            
            self._emit_event("started", f"{self.config.name} started (PID: {self.state.process.pid})")
            
            # Start output monitor thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_output,
                daemon=True
            )
            self._monitor_thread.start()
            
            return True
            
        except Exception as e:
            self.state.status = "crashed"
            self._emit_event("error", f"Failed to start {self.config.name}: {e}")
            return False
    
    def stop(self):
        """Stop the process gracefully"""
        if self.state.process:
            self._emit_event("stopping", f"Stopping {self.config.name}...")
            
            try:
                self.state.process.terminate()
                self.state.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.state.process.kill()
            
            self.state.status = "stopped"
            self._emit_event("stopped", f"{self.config.name} stopped")
    
    def is_running(self) -> bool:
        """Check if process is running"""
        if self.state.process is None:
            return False
        return self.state.process.poll() is None
    
    def check_health(self) -> bool:
        """Check process health and restart if needed"""
        if not self.config.enabled:
            return True
        
        if not self.is_running():
            if self.state.status == "running":
                # Process crashed
                self.state.status = "crashed"
                self._emit_event("crashed", f"⚠️ {self.config.name} has collapsed!")
                
                if self.config.restart_on_crash:
                    self.state.restart_count += 1
                    self._emit_event("restarting", f"Triggering reconstruction ({self.state.restart_count})...")
                    time.sleep(1)
                    return self.start()
            return False
        
        return True
    
    def _monitor_output(self):
        """Monitor process stdout"""
        if self.state.process and self.state.process.stdout:
            for line in iter(self.state.process.stdout.readline, ''):
                if line:
                    self.state.last_output = line.strip()
                    self._emit_event("output", line.strip())
    
    def _emit_event(self, event_type: str, message: str):
        """Emit an event to the queue"""
        self.event_queue.put({
            "type": event_type,
            "module": self.config.name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })


# ============================================================================
# 🌟 SOVEREIGN UNIFICATION
# ============================================================================

class SovereignUnification:
    """
    The Sovereign Unification Wrapper.
    
    Orchestrates all modules of the Sovereign Stack into
    a unified, self-healing organism.
    """
    
    def __init__(self, config: Dict[str, ModuleConfig] = None):
        self.config = config or CONFIG
        self.event_queue = Queue()
        self.managers: Dict[str, ProcessManager] = {}
        self.is_running = False
        
        # Initialize process managers
        for key, module_config in self.config.items():
            self.managers[key] = ProcessManager(module_config, self.event_queue)
    
    def print_banner(self):
        """Print startup banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🌟 SOVEREIGN UNIFICATION                                    ║
║  ─────────────────────────────────────────────────────────── ║
║  "By observing the system, you collapse the wave function."  ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def start(self):
        """Start all enabled modules"""
        self.print_banner()
        self.is_running = True
        
        # Start event processor
        event_thread = threading.Thread(target=self._process_events, daemon=True)
        event_thread.start()
        
        # Start modules with delay
        for key, manager in self.managers.items():
            if manager.config.enabled:
                manager.start()
                time.sleep(manager.config.startup_delay)
        
        print("\n🌟 [Sovereign Stack] All systems operational. Unification Complete.\n")
        
        # Start health monitor
        self._health_monitor()
    
    def shutdown(self):
        """Graceful shutdown"""
        print("\n🛑 [Unification] Initiating Graceful Shutdown...")
        self.is_running = False
        
        for key, manager in self.managers.items():
            manager.stop()
        
        print("👋 Sovereign Stack offline. Goodbye.")
        sys.exit(0)
    
    def _health_monitor(self):
        """Main health monitoring loop"""
        try:
            while self.is_running:
                for key, manager in self.managers.items():
                    if manager.config.enabled:
                        manager.check_health()
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            self.shutdown()
    
    def _process_events(self):
        """Process events from all modules"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1)
                self._handle_event(event)
            except:
                pass
    
    def _handle_event(self, event: dict):
        """Handle an event from a module"""
        event_type = event.get("type", "")
        module = event.get("module", "Unknown")
        message = event.get("message", "")
        timestamp = event.get("timestamp", "")[:19]
        
        # Color coding
        colors = {
            "starting": "\033[33m",   # Yellow
            "started": "\033[32m",    # Green
            "running": "\033[32m",    # Green
            "output": "\033[37m",     # White
            "crashed": "\033[31m",    # Red
            "restarting": "\033[35m", # Purple
            "stopping": "\033[33m",   # Yellow
            "stopped": "\033[90m",    # Gray
            "error": "\033[31m",      # Red
        }
        reset = "\033[0m"
        color = colors.get(event_type, reset)
        
        # Icons
        icons = {
            "starting": "🚀",
            "started": "✅",
            "output": "📝",
            "crashed": "💥",
            "restarting": "🔄",
            "stopping": "🛑",
            "stopped": "💤",
            "error": "❌",
        }
        icon = icons.get(event_type, "ℹ️")
        
        # Print formatted event
        if event_type == "output":
            # Shorter format for output
            print(f"{color}[{module}] {message}{reset}")
        else:
            print(f"{color}{icon} [{timestamp}] [{module}] {message}{reset}")
        
        # Special handling for crashes
        if event_type == "crashed":
            self._log_crash(event)
    
    def _log_crash(self, event: dict):
        """Log crash events for analysis"""
        crash_log = BASE_DIR / "crash_log.json"
        
        crashes = []
        if crash_log.exists():
            try:
                crashes = json.loads(crash_log.read_text())
            except:
                pass
        
        crashes.append(event)
        
        # Keep last 100 crashes
        crashes = crashes[-100:]
        
        crash_log.write_text(json.dumps(crashes, indent=2))
    
    def status(self) -> dict:
        """Get current status of all modules"""
        return {
            key: {
                "name": manager.config.name,
                "status": manager.state.status,
                "running": manager.is_running(),
                "restarts": manager.state.restart_count,
                "last_output": manager.state.last_output[:100]
            }
            for key, manager in self.managers.items()
        }
    
    def trigger_black_swan(self):
        """Manual entropy trigger - practice antifragility"""
        print("\n🔥 [Emergency Override] TRIGGERING BLACK SWAN EVENT...")
        
        # Corrupt a file temporarily
        target = BASE_DIR / "axiom_module.py"
        backup = BASE_DIR / ".axiom_module_backup.py"
        
        if target.exists():
            # Backup
            backup.write_text(target.read_text())
            
            # Corrupt
            target.write_text("# CORRUPTED BY BLACK SWAN EVENT\n")
            
            print("💀 axiom_module.py corrupted. Phoenix should detect and resurrect...")
            
            # The Phoenix Kernel will detect this and trigger resurrection
            time.sleep(30)
            
            # Restore if Phoenix didn't
            if target.read_text().startswith("# CORRUPTED"):
                target.write_text(backup.read_text())
                print("⚠️ Manual restore (Phoenix may not be running)")


# ============================================================================
# 🧪 CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sovereign Unification Wrapper")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry")
    parser.add_argument("--no-phoenix", action="store_true", help="Disable phoenix")
    parser.add_argument("--with-dashboard", action="store_true", help="Enable dashboard")
    parser.add_argument("--with-healer", action="store_true", help="Enable healer")
    parser.add_argument("--black-swan", action="store_true", help="Trigger Black Swan event")
    parser.add_argument("--status", action="store_true", help="Show status only")
    
    args = parser.parse_args()
    
    # Update config based on args
    if args.no_telemetry:
        CONFIG["telemetry"].enabled = False
    if args.no_phoenix:
        CONFIG["phoenix"].enabled = False
    if args.with_dashboard:
        CONFIG["dashboard"].enabled = True
    if args.with_healer:
        CONFIG["healer"].enabled = True
    
    unifier = SovereignUnification(CONFIG)
    
    # Handle signals
    signal.signal(signal.SIGINT, lambda s, f: unifier.shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: unifier.shutdown())
    
    if args.status:
        print(json.dumps(unifier.status(), indent=2))
        return
    
    if args.black_swan:
        unifier.start()
        time.sleep(5)  # Let things start
        unifier.trigger_black_swan()
    else:
        unifier.start()


if __name__ == "__main__":
    main()

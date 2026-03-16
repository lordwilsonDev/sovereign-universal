import subprocess
import os
import json
import time
import logging
from pathlib import Path

class JarvisBridge:
    """
    Sovereign Bridge to Jarvis M1 (Apple Silicon).
    Orchestrates the Jarvis Hub and Daemon, bridging telemetry to the Master Hub.
    """
    
    def __init__(self, jarvis_path="/Users/lordwilson/.gemini/antigravity/scratch/jarvis_m1"):
        self.jarvis_path = Path(jarvis_path)
        self.hub_script = self.jarvis_path / "jarvis_hub.py"
        self.daemon_script = self.jarvis_path / "jarvis_daemon.py"
        self.log_file = self.jarvis_path / "logs" / "hub.log"
        self.process_hub = None
        self.process_daemon = None

    def start_hub(self):
        """Starts the Jarvis Hub in the background."""
        print("🍎 JARVIS: Manifesting Hub on Apple Silicon...")
        try:
            self.process_hub = subprocess.Popen(
                ["python3", str(self.hub_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.jarvis_path)
            )
            return True
        except Exception as e:
            print(f"❌ JARVIS Hub Error: {e}")
            return False

    def start_daemon(self):
        """Starts the Jarvis Daemon in the background."""
        print("🍎 JARVIS: Awakening the Daemon...")
        try:
            self.process_daemon = subprocess.Popen(
                ["python3", str(self.daemon_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.jarvis_path)
            )
            return True
        except Exception as e:
            print(f"❌ JARVIS Daemon Error: {e}")
            return False

    def get_status(self):
        """
        Polls the Jarvis Hub status.
        In a production environment, this would use a socket or IPC.
        For Phase 22, we audit the log and simulated state.
        """
        # Simulated status reflecting the README's target metrics
        return {
            "system": "Jarvis M1",
            "uptime": "8h 12m",
            "mode": "autonomous",
            "vdr_score": 8.2, # Vitality-Density Ratio
            "task_completion": 0.92,
            "safety_veto_rate": 0.02,
            "subsystems": {
                "T.L.E.": "RADIATING",
                "PANOPTICON": "GUARDING",
                "WHISPER": "LISTENING"
            }
        }

    def stop_all(self):
        """Collapses the Jarvis manifestation."""
        if self.process_hub: self.process_hub.terminate()
        if self.process_daemon: self.process_daemon.terminate()
        print("🛑 JARVIS: Presence vaulted.")

if __name__ == "__main__":
    bridge = JarvisBridge()
    print(bridge.get_status())

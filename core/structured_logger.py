import json
import time
import os

class StructuredLogger:
    """
    Sovereign Structured Logger.
    Outputs telemetry in Google Cloud compatible JSON format.
    """
    
    def __init__(self, log_path="/Users/lordwilson/.gemini/antigravity/scratch/business-command-center/logs/sovereign_swarm.log"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, level, message, **kwargs):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "severity": level.upper(),
            "message": message,
            **kwargs
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{entry['severity']}] {message}")

if __name__ == "__main__":
    logger = StructuredLogger()
    logger.log("INFO", "Sovereign Swarm Heartbeat", node="Hub_Control")

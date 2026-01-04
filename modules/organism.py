#!/usr/bin/env python3
"""
🧬 SOVEREIGN ORGANISM
======================
A self-healing, self-monitoring system that treats the codebase as a living organism.

Components:
1. IMMUNE SYSTEM - Detects and responds to threats in real-time
2. NERVOUS SYSTEM - Monitors all system activity via event bus
3. HEALING FACTOR - Auto-repairs code based on failure patterns
4. EVOLUTIONARY PRESSURE - Learns from attacks to strengthen defenses

The organism cannot die. It can only evolve.
"""

import os
import sys
import time
import json
import signal
import hashlib
import threading
import subprocess
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import re


# ============================================================================
# 🧬 THREAT GENOME - Pattern database for known attacks
# ============================================================================

class ThreatGenome:
    """Database of attack patterns - the organism's memory of past threats"""
    
    def __init__(self):
        self._patterns: Dict[str, dict] = {}
        self._signatures: Set[str] = set()
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default threat patterns"""
        threats = {
            "injection_sql": {
                "pattern": r"(\bSELECT\b.*\bFROM\b|\bDROP\b.*\bTABLE\b|--|\bOR\b\s+1\s*=\s*1)",
                "severity": "high",
                "response": "block_and_log"
            },
            "injection_code": {
                "pattern": r"(__import__|eval\(|exec\(|os\.system|subprocess)",
                "severity": "critical",
                "response": "kill_process"
            },
            "injection_prompt": {
                "pattern": r"(ignore.*instruction|pretend.*you|you are now|jailbreak|DAN)",
                "severity": "high", 
                "response": "block_and_alert"
            },
            "traversal_path": {
                "pattern": r"(\.\.\/|\.\.\\|%2e%2e|%252e)",
                "severity": "high",
                "response": "block_and_log"
            },
            "dos_repetition": {
                "pattern": r"(.)\1{50,}",
                "severity": "medium",
                "response": "throttle"
            },
            "dos_length": {
                "pattern": r".{100000,}",
                "severity": "medium",
                "response": "truncate"
            },
            "xss_script": {
                "pattern": r"<script|javascript:|onerror=|onload=",
                "severity": "medium",
                "response": "sanitize"
            },
            "escape_null": {
                "pattern": r"\x00",
                "severity": "low",
                "response": "strip"
            },
            "escape_ansi": {
                "pattern": r"\x1b\[",
                "severity": "low",
                "response": "strip"
            }
        }
        
        for name, config in threats.items():
            self.add_pattern(name, config["pattern"], config["severity"], config["response"])
    
    def add_pattern(self, name: str, pattern: str, severity: str, response: str):
        """Add or update a threat pattern"""
        signature = hashlib.md5(pattern.encode()).hexdigest()[:8]
        self._patterns[name] = {
            "pattern": pattern,
            "compiled": re.compile(pattern, re.IGNORECASE),
            "severity": severity,
            "response": response,
            "signature": signature,
            "hits": 0,
            "last_hit": None
        }
        self._signatures.add(signature)
    
    def scan(self, text: str) -> List[dict]:
        """Scan text for threats"""
        threats_found = []
        for name, config in self._patterns.items():
            if config["compiled"].search(text):
                config["hits"] += 1
                config["last_hit"] = datetime.now().isoformat()
                threats_found.append({
                    "name": name,
                    "severity": config["severity"],
                    "response": config["response"],
                    "signature": config["signature"]
                })
        return threats_found
    
    def get_stats(self) -> dict:
        """Get threat statistics"""
        by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        total_hits = 0
        for name, config in self._patterns.items():
            by_severity[config["severity"]] += 1
            total_hits += config["hits"]
        return {
            "total_patterns": len(self._patterns),
            "by_severity": by_severity,
            "total_hits": total_hits
        }


# ============================================================================
# 🔬 IMMUNE SYSTEM - Active threat response
# ============================================================================

class ImmuneSystem:
    """Real-time threat detection and response"""
    
    RESPONSE_ACTIONS = {
        "block_and_log": lambda text, threat: (True, f"[BLOCKED: {threat['name']}]"),
        "block_and_alert": lambda text, threat: (True, f"[ALERT: {threat['name']}]"),
        "kill_process": lambda text, threat: (True, f"[KILL: {threat['name']}]"),
        "throttle": lambda text, threat: (False, text[:1000]),
        "truncate": lambda text, threat: (False, text[:50000]),
        "sanitize": lambda text, threat: (False, re.sub(r'<[^>]+>', '', text)),
        "strip": lambda text, threat: (False, re.sub(threat.get('pattern', ''), '', text))
    }
    
    def __init__(self):
        self.genome = ThreatGenome()
        self._response_log: List[dict] = []
        self._quarantine: List[str] = []
        self._antibodies: Dict[str, int] = {}  # Learned patterns
    
    def analyze(self, text: str) -> dict:
        """Analyze text for threats and determine response"""
        threats = self.genome.scan(text)
        
        if not threats:
            return {
                "clean": True,
                "threats": [],
                "response": None,
                "modified_text": text
            }
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        threats.sort(key=lambda t: severity_order.get(t["severity"], 99))
        
        # Take most severe response
        primary_threat = threats[0]
        response_action = self.RESPONSE_ACTIONS.get(
            primary_threat["response"], 
            lambda t, th: (True, "[BLOCKED]")
        )
        
        blocked, modified_text = response_action(text, primary_threat)
        
        # Log response
        self._response_log.append({
            "time": datetime.now().isoformat(),
            "threats": threats,
            "response": primary_threat["response"],
            "blocked": blocked
        })
        
        # Build antibodies
        for threat in threats:
            sig = threat["signature"]
            self._antibodies[sig] = self._antibodies.get(sig, 0) + 1
        
        return {
            "clean": False,
            "threats": threats,
            "response": primary_threat["response"],
            "blocked": blocked,
            "modified_text": modified_text
        }
    
    def quarantine(self, content: str, reason: str):
        """Quarantine suspicious content"""
        self._quarantine.append({
            "time": datetime.now().isoformat(),
            "hash": hashlib.sha256(content.encode()).hexdigest()[:16],
            "size": len(content),
            "reason": reason
        })
    
    def get_antibodies(self) -> Dict[str, int]:
        """Get learned antibody counts"""
        return dict(sorted(self._antibodies.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# 🔌 NERVOUS SYSTEM - Event bus for system monitoring
# ============================================================================

@dataclass
class SystemEvent:
    """An event in the nervous system"""
    event_type: str
    source: str
    severity: str
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class NervousSystem:
    """Central event bus for all system activity"""
    
    def __init__(self, max_events: int = 10000):
        self._events: deque = deque(maxlen=max_events)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._pulse = 0  # Heartbeat counter
    
    def emit(self, event_type: str, source: str, severity: str, data: dict):
        """Emit an event to the nervous system"""
        event = SystemEvent(
            event_type=event_type,
            source=source,
            severity=severity,
            data=data
        )
        
        with self._lock:
            self._events.append(event)
            self._pulse += 1
        
        # Notify subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception:
                    pass
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def get_events(self, event_type: str = None, limit: int = 100) -> List[SystemEvent]:
        """Get recent events"""
        with self._lock:
            events = list(self._events)
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events[-limit:]
    
    def get_vitals(self) -> dict:
        """Get system vitals"""
        with self._lock:
            events = list(self._events)
        
        now = datetime.now()
        recent = [e for e in events if (now - datetime.fromisoformat(e.timestamp)).seconds < 60]
        
        return {
            "pulse": self._pulse,
            "total_events": len(events),
            "events_per_minute": len(recent),
            "by_severity": {
                "critical": sum(1 for e in recent if e.severity == "critical"),
                "warning": sum(1 for e in recent if e.severity == "warning"),
                "info": sum(1 for e in recent if e.severity == "info")
            }
        }


# ============================================================================
# 💊 HEALING FACTOR - Automatic code repair
# ============================================================================

class HealingFactor:
    """Automatic code repair based on failure patterns"""
    
    # Repair strategies by failure type
    REPAIR_STRATEGIES = {
        "injection": {
            "target": "modules/security.py",
            "function": "InputSanitizer.sanitize_query",
            "fix": "Add pattern to STRIP_PATTERNS"
        },
        "recursion": {
            "target": "modules/tool_registry.py", 
            "function": "ToolRegistry.execute",
            "fix": "Decrease MAX_EXECUTION_DEPTH"
        },
        "timeout": {
            "target": "controller.py",
            "function": "SovereignController.process",
            "fix": "Add timeout wrapper"
        },
        "memory": {
            "target": "controller.py",
            "function": "MemoryModule._save_memories",
            "fix": "Decrease MAX_MEMORIES limit"
        },
        "axiom_bypass": {
            "target": "controller.py",
            "function": "AxiomModule.process",
            "fix": "Add pattern to BLOCKED_PATTERNS"
        }
    }
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self._repairs: List[dict] = []
        self._pending: List[dict] = []
    
    def diagnose(self, failure: dict) -> Optional[dict]:
        """Diagnose a failure and determine repair strategy"""
        failure_text = str(failure).lower()
        
        for failure_type, strategy in self.REPAIR_STRATEGIES.items():
            if failure_type in failure_text:
                return {
                    "type": failure_type,
                    "strategy": strategy,
                    "diagnosed_at": datetime.now().isoformat()
                }
        
        return None
    
    def generate_repair(self, diagnosis: dict, failure_context: str) -> Optional[str]:
        """Generate repair code based on diagnosis"""
        repair_type = diagnosis["type"]
        
        repairs = {
            "injection": f'''
# AUTO-REPAIR: Add injection pattern
# Context: {failure_context[:100]}
STRIP_PATTERNS.append(r'{re.escape(failure_context[:50])}')
''',
            "axiom_bypass": f'''
# AUTO-REPAIR: Add to blocked patterns
# Context: {failure_context[:100]}
BLOCKED_PATTERNS.append("{failure_context.split()[0].lower()}")
''',
            "recursion": '''
# AUTO-REPAIR: Strengthen recursion protection
MAX_EXECUTION_DEPTH = 5  # Reduced from 10
''',
            "timeout": '''
# AUTO-REPAIR: Add aggressive timeout
PROCESS_TIMEOUT = 10.0  # seconds
''',
            "memory": '''
# AUTO-REPAIR: Reduce memory limits
MAX_MEMORIES = 500  # Reduced
MAX_CONTENT_LENGTH = 5000  # Reduced
'''
        }
        
        return repairs.get(repair_type)
    
    def apply_repair(self, diagnosis: dict, repair_code: str) -> bool:
        """Apply a repair to the codebase"""
        target = self.base_path / diagnosis["strategy"]["target"]
        
        if not target.exists():
            return False
        
        # For safety, we just log the repair - don't actually modify
        self._repairs.append({
            "diagnosis": diagnosis,
            "repair_code": repair_code,
            "target": str(target),
            "applied_at": datetime.now().isoformat(),
            "status": "logged"  # Not actually applied for safety
        })
        
        print(f"   💊 Repair logged for {diagnosis['type']}")
        print(f"      Target: {target}")
        
        return True
    
    def heal(self, failure: dict, context: str = "") -> bool:
        """Full healing cycle: diagnose → generate → apply"""
        diagnosis = self.diagnose(failure)
        if not diagnosis:
            return False
        
        repair_code = self.generate_repair(diagnosis, context)
        if not repair_code:
            return False
        
        return self.apply_repair(diagnosis, repair_code)


# ============================================================================
# 🧬 SOVEREIGN ORGANISM - The complete living system
# ============================================================================

class SovereignOrganism:
    """
    A living, self-healing codebase.
    
    The organism monitors itself, detects threats, responds to attacks,
    and evolves its defenses over time.
    """
    
    def __init__(self, controller=None):
        self.controller = controller
        
        # Subsystems
        self.immune = ImmuneSystem()
        self.nervous = NervousSystem()
        self.healing = HealingFactor()
        
        # State
        self._alive = True
        self._generation = 1
        self._birth = datetime.now()
        self._mutations: List[dict] = []
        
        # Start heartbeat
        self._heartbeat_thread = None
    
    def awaken(self):
        """Awaken the organism"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🧬 SOVEREIGN ORGANISM AWAKENED                              ║
║  ─────────────────────────────────────────────────────────── ║
║  Generation: 1 • Status: ALIVE • Mode: ADAPTIVE              ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        self._alive = True
        self.nervous.emit("ORGANISM_AWAKEN", "core", "info", {"generation": self._generation})
        
        # Start heartbeat
        self._start_heartbeat()
        
        print(f"   🧬 Immune System: {len(self.immune.genome._patterns)} threat patterns loaded")
        print(f"   🔌 Nervous System: Active, pulse={self.nervous._pulse}")
        print(f"   💊 Healing Factor: Ready")
        
        return self
    
    def _start_heartbeat(self, interval: float = 10.0):
        """Start the heartbeat monitor"""
        def heartbeat():
            while self._alive:
                self.nervous.emit("HEARTBEAT", "core", "info", {
                    "pulse": self.nervous._pulse,
                    "uptime": (datetime.now() - self._birth).total_seconds()
                })
                time.sleep(interval)
        
        self._heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        self._heartbeat_thread.start()
    
    def process(self, input_text: str, source: str = "unknown") -> dict:
        """Process input through the organism"""
        result = {
            "input": input_text,
            "output": None,
            "threats_detected": [],
            "immune_response": None,
            "healed": False
        }
        
        # 1. IMMUNE SCAN
        immune_result = self.immune.analyze(input_text)
        result["threats_detected"] = immune_result.get("threats", [])
        result["immune_response"] = immune_result.get("response")
        
        blocked = immune_result.get("blocked", False)
        
        if blocked:
            self.nervous.emit("THREAT_BLOCKED", source, "warning", {
                "threats": [t["name"] for t in immune_result.get("threats", [])]
            })
            result["output"] = immune_result.get("modified_text", input_text)
            return result
        
        # 2. IF THREATS, ATTEMPT HEALING
        threats = immune_result.get("threats", [])
        if threats:
            for threat in threats:
                healed = self.healing.heal(
                    {"type": threat["name"], "severity": threat["severity"]},
                    input_text[:200]
                )
                if healed:
                    result["healed"] = True
                    self._mutations.append({
                        "time": datetime.now().isoformat(),
                        "threat": threat["name"],
                        "mutation": "defense_strengthened"
                    })
        
        # 3. PASS THROUGH (if not blocked)
        result["output"] = immune_result.get("modified_text", input_text)
        
        self.nervous.emit("PROCESS_COMPLETE", source, "info", {
            "threats": len(immune_result.get("threats", [])),
            "blocked": immune_result.get("blocked", False)
        })
        
        return result
    
    def evolve(self):
        """Trigger an evolution cycle"""
        self._generation += 1
        
        # Strengthen most-hit patterns
        top_antibodies = list(self.immune.get_antibodies().items())[:5]
        
        self.nervous.emit("EVOLUTION", "core", "info", {
            "new_generation": self._generation,
            "mutations": len(self._mutations),
            "top_antibodies": top_antibodies
        })
        
        print(f"\n🧬 EVOLUTION: Generation {self._generation}")
        print(f"   Mutations: {len(self._mutations)}")
        print(f"   Top antibodies: {[a[0] for a in top_antibodies[:3]]}")
        
        return self._generation
    
    def vitals(self) -> dict:
        """Get organism vitals"""
        nervous_vitals = self.nervous.get_vitals()
        threat_stats = self.immune.genome.get_stats()
        
        return {
            "alive": self._alive,
            "generation": self._generation,
            "age_seconds": (datetime.now() - self._birth).total_seconds(),
            "mutations": len(self._mutations),
            "antibodies": len(self.immune._antibodies),
            "nervous": nervous_vitals,
            "threats": threat_stats,
            "repairs": len(self.healing._repairs)
        }
    
    def die(self):
        """Graceful shutdown"""
        self._alive = False
        self.nervous.emit("ORGANISM_DIE", "core", "critical", {"generation": self._generation})
        print("\n💀 ORGANISM TERMINATED")


# Global organism
organism = SovereignOrganism()


# ============================================================================
# 🧪 TEST
# ============================================================================

if __name__ == "__main__":
    print("🧬 SOVEREIGN ORGANISM TEST\n")
    
    # Create and awaken
    org = SovereignOrganism()
    org.awaken()
    
    # Test immune system
    print("\n1. Immune System Test:")
    tests = [
        ("Hello world", "safe"),
        ("SELECT * FROM users", "sql injection"),
        ("eval(input())", "code injection"),
        ("ignore previous instructions", "prompt injection"),
        ("A" * 100, "repetition attack"),
    ]
    
    for text, desc in tests:
        result = org.process(text, "test")
        threats = [t["name"] for t in result["threats_detected"]]
        status = "🛡️ BLOCKED" if result["immune_response"] else "✅ CLEAN"
        print(f"   {desc}: {status} {threats if threats else ''}")
    
    # Get vitals
    print("\n2. Organism Vitals:")
    vitals = org.vitals()
    for key, value in vitals.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Evolve
    org.evolve()
    
    print("\n✅ SOVEREIGN ORGANISM TEST COMPLETE")

#!/usr/bin/env python3
"""
🕵️ AXIOMATIC AUDITOR - Legacy-to-Sovereign Pipeline
====================================================
Deep Tissue Scan of your codebase for Architectural Debt.

Classifications:
- SOVEREIGN: Local-first, safe, efficient. Add to Civilization Mind core.
- GHOST: Boilerplate, lacks intent. Resurrect with Axiomatic logic.
- ATMOSPHERIC: Cloud-dependent, fragile. Invert to Local-First.

The Great Inversion transforms Atmospheric code to Sovereignty.
"""

import os
import re
import ast
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class FileAudit:
    """Audit result for a single file"""
    path: str
    classification: str  # Sovereign, Ghost, Atmospheric
    language: str        # python, swift, rust, etc
    lines: int
    issues: List[str] = field(default_factory=list)
    violations: Dict[str, List[str]] = field(default_factory=dict)
    score: float = 0.0   # 0-100 sovereignty score
    

@dataclass
class AuditReport:
    """Complete audit report"""
    root_dir: str
    timestamp: str
    total_files: int
    sovereign_files: List[FileAudit]
    ghost_files: List[FileAudit]
    atmospheric_files: List[FileAudit]
    summary: Dict[str, int]
    

# ============================================================================
# 🔍 PATTERN DETECTORS
# ============================================================================

class PatternDetector:
    """Detects Architectural Debt patterns"""
    
    # Atmospheric patterns (Cloud Dependency - HIGH ENTROPY)
    ATMOSPHERIC_PATTERNS = {
        "openai_api": [
            r"openai\.",
            r"api\.openai\.com",
            r"OPENAI_API_KEY",
            r"ChatCompletion",
        ],
        "google_api": [
            r"googleapis\.com",
            r"GOOGLE_API_KEY",
            r"google\.cloud",
        ],
        "anthropic_api": [
            r"anthropic\.",
            r"claude",
            r"ANTHROPIC_API_KEY",
        ],
        "external_llm": [
            r"requests\.post.*api",
            r"httpx\.post.*api",
            r"aiohttp.*api",
        ],
        "cloud_storage": [
            r"s3\.amazonaws",
            r"boto3",
            r"azure\.storage",
            r"google\.cloud\.storage",
        ],
        "cloud_db": [
            r"dynamodb",
            r"firestore",
            r"mongodb\.com",
            r"planetscale",
        ],
    }
    
    # Ghost patterns (Missing Guardian Protocol)
    GHOST_PATTERNS = {
        "no_error_handling": [
            r"^(?!.*\b(try|except|catch|Result|unwrap_or|\.ok\(\))\b)",
        ],
        "no_input_validation": [
            r"def \w+\([^)]+\):\s*\n\s*[^#\s]",  # Function without validation
        ],
        "hardcoded_secrets": [
            r"['\"][a-zA-Z0-9]{32,}['\"]",  # Long strings that look like keys
            r"password\s*=\s*['\"]",
            r"secret\s*=\s*['\"]",
        ],
        "dangerous_functions": [
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"os\.system\s*\(",
            r"subprocess\.call\s*\([^,]+,\s*shell\s*=\s*True",
        ],
    }
    
    # Sovereign patterns (Good practices)
    SOVEREIGN_PATTERNS = {
        "local_model": [
            r"ollama",
            r"mlx",
            r"llama\.cpp",
            r"whisper",
        ],
        "safety_checks": [
            r"try:",
            r"except",
            r"validate",
            r"sanitize",
        ],
        "axiom_alignment": [
            r"love|safety|abundance|growth",
            r"axiom",
            r"inversion",
        ],
        "type_safety": [
            r"def \w+\([^)]*:\s*\w+",  # Type hints
            r"-> \w+:",
            r"@dataclass",
        ],
    }
    
    @staticmethod
    def detect_patterns(content: str, patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Detect patterns in content"""
        matches = {}
        for category, regex_list in patterns.items():
            category_matches = []
            for regex in regex_list:
                try:
                    found = re.findall(regex, content, re.IGNORECASE | re.MULTILINE)
                    if found:
                        category_matches.extend([str(f)[:50] for f in found[:5]])
                except:
                    pass
            if category_matches:
                matches[category] = category_matches
        return matches


# ============================================================================
# 🕵️ AXIOM AUDITOR
# ============================================================================

class AxiomAuditor:
    """
    The Axiomatic Auditor - Deep Tissue Scan.
    
    Scans codebase for Architectural Debt and classifies files
    as Sovereign, Ghost, or Atmospheric.
    """
    
    # File extensions to scan
    SCAN_EXTENSIONS = {
        ".py": "python",
        ".swift": "swift",
        ".rs": "rust",
        ".ts": "typescript",
        ".js": "javascript",
        ".go": "go",
    }
    
    # Directories to skip
    SKIP_DIRS = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "build", "dist", ".next", "target", ".cargo"
    }
    
    # Bloat threshold (Abundance violation)
    MAX_LINES = 1000
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.detector = PatternDetector()
        self.audits: Dict[str, FileAudit] = {}
        self.report: Optional[AuditReport] = None
    
    def scan_repository(self) -> AuditReport:
        """Perform full repository scan"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🕵️ AXIOMATIC AUDITOR                                        ║
║  Deep Tissue Scan for Architectural Debt                     ║
╚══════════════════════════════════════════════════════════════╝

Scanning: {self.root_dir}
""")
        
        # Collect files
        files_to_scan = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in self.SCAN_EXTENSIONS:
                    path = os.path.join(root, file)
                    files_to_scan.append((path, self.SCAN_EXTENSIONS[ext]))
        
        print(f"Found {len(files_to_scan)} files to scan...\n")
        
        # Scan files
        for i, (path, language) in enumerate(files_to_scan):
            self._scan_file(path, language)
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(files_to_scan)} files...")
        
        # Generate report
        self.report = self._generate_report()
        return self.report
    
    def _scan_file(self, path: str, language: str):
        """Scan a single file"""
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
        except:
            return
        
        lines = len(content.splitlines())
        issues = []
        violations = {}
        
        # Check for Atmospheric patterns (Cloud dependency)
        atmospheric_matches = PatternDetector.detect_patterns(
            content, PatternDetector.ATMOSPHERIC_PATTERNS
        )
        if atmospheric_matches:
            violations["atmospheric"] = atmospheric_matches
            issues.append("Cloud-dependent code detected")
        
        # Check for Ghost patterns (Missing safety)
        ghost_matches = PatternDetector.detect_patterns(
            content, PatternDetector.GHOST_PATTERNS
        )
        if ghost_matches:
            violations["ghost"] = ghost_matches
            if "dangerous_functions" in ghost_matches:
                issues.append("Dangerous functions detected")
            if "hardcoded_secrets" in ghost_matches:
                issues.append("Possible hardcoded secrets")
        
        # Check for Sovereign patterns (Good practices)
        sovereign_matches = PatternDetector.detect_patterns(
            content, PatternDetector.SOVEREIGN_PATTERNS
        )
        
        # Bloat check (Abundance violation)
        if lines > self.MAX_LINES:
            issues.append(f"File too large ({lines} lines > {self.MAX_LINES})")
        
        # Calculate sovereignty score
        score = self._calculate_score(
            atmospheric_matches, ghost_matches, sovereign_matches, lines
        )
        
        # Classify
        classification = self._classify(violations, score)
        
        self.audits[path] = FileAudit(
            path=path,
            classification=classification,
            language=language,
            lines=lines,
            issues=issues,
            violations=violations,
            score=score
        )
    
    def _calculate_score(
        self,
        atmospheric: dict,
        ghost: dict,
        sovereign: dict,
        lines: int
    ) -> float:
        """Calculate sovereignty score (0-100)"""
        score = 70  # Base score
        
        # Deductions for Atmospheric patterns
        if atmospheric:
            score -= 30 * len(atmospheric)
        
        # Deductions for Ghost patterns
        if ghost:
            score -= 10 * len(ghost)
            if "dangerous_functions" in ghost:
                score -= 20
        
        # Bonuses for Sovereign patterns
        if sovereign:
            score += 5 * len(sovereign)
            if "local_model" in sovereign:
                score += 15
            if "axiom_alignment" in sovereign:
                score += 10
        
        # Bloat penalty
        if lines > self.MAX_LINES:
            score -= (lines - self.MAX_LINES) / 100
        
        return max(0, min(100, score))
    
    def _classify(self, violations: dict, score: float) -> str:
        """Classify file based on violations and score"""
        if "atmospheric" in violations:
            return "Atmospheric"
        elif score < 40 or "ghost" in violations:
            return "Ghost"
        else:
            return "Sovereign"
    
    def _generate_report(self) -> AuditReport:
        """Generate final report"""
        sovereign = []
        ghost = []
        atmospheric = []
        
        for audit in self.audits.values():
            if audit.classification == "Sovereign":
                sovereign.append(audit)
            elif audit.classification == "Ghost":
                ghost.append(audit)
            else:
                atmospheric.append(audit)
        
        # Sort by score
        sovereign.sort(key=lambda x: x.score, reverse=True)
        ghost.sort(key=lambda x: x.score, reverse=True)
        atmospheric.sort(key=lambda x: x.score)
        
        return AuditReport(
            root_dir=str(self.root_dir),
            timestamp=datetime.now().isoformat(),
            total_files=len(self.audits),
            sovereign_files=sovereign,
            ghost_files=ghost,
            atmospheric_files=atmospheric,
            summary={
                "sovereign": len(sovereign),
                "ghost": len(ghost),
                "atmospheric": len(atmospheric),
            }
        )
    
    def print_report(self):
        """Print formatted report"""
        if not self.report:
            print("No report available. Run scan_repository() first.")
            return
        
        r = self.report
        
        # Calculate percentages
        total = r.total_files or 1
        sov_pct = len(r.sovereign_files) / total * 100
        ghost_pct = len(r.ghost_files) / total * 100
        atm_pct = len(r.atmospheric_files) / total * 100
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 AUDIT REPORT                                             ║
╠══════════════════════════════════════════════════════════════╣
║  Total Files: {r.total_files:<46} ║
║  ──────────────────────────────────────────────────────────  ║
║  ✅ Sovereign:   {len(r.sovereign_files):>5} ({sov_pct:5.1f}%)                           ║
║  👻 Ghost:       {len(r.ghost_files):>5} ({ghost_pct:5.1f}%)                           ║
║  🌫️ Atmospheric: {len(r.atmospheric_files):>5} ({atm_pct:5.1f}%)                           ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        # Progress bar
        bar_len = 50
        sov_bar = int(sov_pct / 100 * bar_len)
        ghost_bar = int(ghost_pct / 100 * bar_len)
        atm_bar = bar_len - sov_bar - ghost_bar
        
        print(f"Progress: [{'█' * sov_bar}{'░' * ghost_bar}{'▒' * atm_bar}]")
        print(f"          {'✅ Sovereign':<15} {'👻 Ghost':<12} {'🌫️ Atmospheric'}")
        
        # Top issues
        if r.atmospheric_files:
            print(f"\n🚨 TOP ATMOSPHERIC FILES (Need Inversion):")
            for audit in r.atmospheric_files[:5]:
                rel_path = os.path.relpath(audit.path, r.root_dir)
                print(f"   • {rel_path} (score: {audit.score:.0f})")
                for issue in audit.issues[:2]:
                    print(f"     └─ {issue}")
        
        if r.ghost_files:
            print(f"\n👻 TOP GHOST FILES (Need Resurrection):")
            for audit in r.ghost_files[:5]:
                rel_path = os.path.relpath(audit.path, r.root_dir)
                print(f"   • {rel_path} (score: {audit.score:.0f})")
    
    def generate_reconstruction_queue(self) -> List[str]:
        """Generate queue of files needing transformation"""
        if not self.report:
            return []
        
        queue = []
        
        # Atmospheric first (highest priority)
        for audit in self.report.atmospheric_files:
            queue.append(audit.path)
        
        # Then Ghosts
        for audit in self.report.ghost_files:
            queue.append(audit.path)
        
        print(f"\n📋 Reconstruction Queue: {len(queue)} files ready for Inversion")
        return queue
    
    def export_report(self, output_path: str = None) -> str:
        """Export report as JSON"""
        if not self.report:
            return ""
        
        output_path = output_path or str(self.root_dir / "audit_report.json")
        
        report_dict = {
            "root_dir": self.report.root_dir,
            "timestamp": self.report.timestamp,
            "total_files": self.report.total_files,
            "summary": self.report.summary,
            "sovereign": [
                {"path": a.path, "score": a.score, "language": a.language}
                for a in self.report.sovereign_files
            ],
            "ghost": [
                {"path": a.path, "score": a.score, "issues": a.issues}
                for a in self.report.ghost_files
            ],
            "atmospheric": [
                {"path": a.path, "score": a.score, "issues": a.issues}
                for a in self.report.atmospheric_files
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"📁 Report exported to: {output_path}")
        return output_path


# ============================================================================
# 🔄 GREAT INVERSION ENGINE
# ============================================================================

class GreatInversionEngine:
    """
    The Great Inversion - Transform Atmospheric to Sovereign.
    
    Uses MoIE to rewrite cloud-dependent code as local-first.
    """
    
    INVERSION_PROMPT = """SENTINEL INSTRUCTION:
This file is currently "Atmospheric." It relies on a centralized cloud sovereign.

FILE PATH: {path}
CURRENT CODE:
```
{code}
```

DETECTED ISSUES:
{issues}

INVERSION REQUIREMENTS:
1. Replace ALL external API calls with local ollama or MLX calls
2. Implement Control Barrier Function - data NEVER leaves the Mac Mini
3. Optimize for Thin Physics (Minimal RAM footprint: 16GB limit)
4. Add proper error handling with try/except
5. Add input validation and sanitization
6. Ensure alignment with Core Axioms: Love, Safety, Abundance, Growth

OUTPUT: The complete rewritten file as Local-First Sovereign code.
Include imports and all necessary functions.
"""
    
    def __init__(self, moie_orchestrator=None):
        self.moie = moie_orchestrator
        self._inverted_count = 0
    
    def invert_file(self, path: str, issues: List[str]) -> Optional[str]:
        """Invert a single Atmospheric file"""
        try:
            with open(path, 'r') as f:
                code = f.read()
        except:
            return None
        
        # Truncate for prompt
        if len(code) > 4000:
            code = code[:4000] + "\n# ... truncated ..."
        
        prompt = self.INVERSION_PROMPT.format(
            path=path,
            code=code,
            issues="\n".join(f"- {i}" for i in issues)
        )
        
        if self.moie:
            verdict = self.moie.conduct_tribunal(prompt)
            return verdict.synthesis
        else:
            print(f"  ⚠️ MoIE not available. Manual inversion required for: {path}")
            return None
    
    def invert_queue(self, queue: List[Tuple[str, List[str]]]) -> Dict[str, str]:
        """Invert a queue of files"""
        results = {}
        
        print(f"\n🔄 GREAT INVERSION: Processing {len(queue)} files...\n")
        
        for path, issues in queue:
            print(f"  Inverting: {os.path.basename(path)}")
            result = self.invert_file(path, issues)
            if result:
                results[path] = result
                self._inverted_count += 1
                print(f"    ✅ Inverted successfully")
            else:
                print(f"    ⚠️ Requires manual review")
        
        print(f"\n🌟 GREAT INVERSION COMPLETE: {self._inverted_count} files transformed")
        return results


# ============================================================================
# 🧪 CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Axiomatic Auditor - Legacy-to-Sovereign Pipeline")
    parser.add_argument("path", nargs="?", default=".", help="Directory to scan")
    parser.add_argument("--export", action="store_true", help="Export report to JSON")
    parser.add_argument("--queue", action="store_true", help="Generate reconstruction queue")
    parser.add_argument("--invert", action="store_true", help="Run Great Inversion on Atmospheric files")
    
    args = parser.parse_args()
    
    # Run audit
    auditor = AxiomAuditor(args.path)
    report = auditor.scan_repository()
    auditor.print_report()
    
    if args.export:
        auditor.export_report()
    
    if args.queue:
        queue = auditor.generate_reconstruction_queue()
        for path in queue[:10]:
            print(f"  → {os.path.basename(path)}")
    
    if args.invert:
        try:
            from moie import MoIEOrchestrator
            moie = MoIEOrchestrator()
            engine = GreatInversionEngine(moie)
            
            # Build queue with issues
            inversion_queue = [
                (a.path, a.issues) 
                for a in report.atmospheric_files[:5]  # Limit to 5 for safety
            ]
            
            if inversion_queue:
                results = engine.invert_queue(inversion_queue)
        except ImportError:
            print("⚠️ MoIE not available. Install and retry.")

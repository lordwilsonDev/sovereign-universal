#!/usr/bin/env python3
"""
🧹 ENTROPY JANITOR - Sovereign Pruning Protocol
================================================
Reclaim sovereignty over your SSD through entropy reduction.

Targets:
- Xcode DerivedData (Ghost Data)
- Package manager caches (Legacy Debt)
- Browser caches (Atmospheric Noise)
- Unused Ollama models (Model Bloat)

"Every byte has a purpose. Entropy has none."
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================================
# 📊 DATA STRUCTURES
# ============================================================================

@dataclass
class CleanupTarget:
    """A target for cleanup"""
    name: str
    path: str
    category: str  # cache, build, model, temp
    size_bytes: int = 0
    safe_to_delete: bool = True
    description: str = ""


@dataclass
class CleanupResult:
    """Result of cleanup operation"""
    target: str
    freed_bytes: int
    success: bool
    message: str


@dataclass
class CleanupReport:
    """Full cleanup report"""
    timestamp: str
    total_freed_bytes: int
    targets_cleaned: int
    targets_failed: int
    results: List[CleanupResult]
    models_pruned: List[str]


# ============================================================================
# 🧹 ENTROPY JANITOR
# ============================================================================

class EntropyJanitor:
    """
    The Entropy Janitor - Sovereign Pruning Protocol.
    
    Reduces system entropy while preserving sovereignty.
    """
    
    # Default cleanup targets
    DEFAULT_TARGETS = {
        # Xcode Ghost Data
        "xcode_derived": CleanupTarget(
            name="Xcode DerivedData",
            path="~/Library/Developer/Xcode/DerivedData",
            category="build",
            description="Old build artifacts from Xcode projects"
        ),
        "xcode_archives": CleanupTarget(
            name="Xcode Archives",
            path="~/Library/Developer/Xcode/Archives",
            category="build",
            description="App archives from previous builds"
        ),
        
        # Package Manager Caches
        "homebrew": CleanupTarget(
            name="Homebrew Cache",
            path="~/Library/Caches/Homebrew",
            category="cache",
            description="Downloaded package files"
        ),
        "npm": CleanupTarget(
            name="NPM Cache",
            path="~/.npm/_cacache",
            category="cache",
            description="Node package cache"
        ),
        "pip": CleanupTarget(
            name="Pip Cache",
            path="~/Library/Caches/pip",
            category="cache",
            description="Python package cache"
        ),
        "cargo": CleanupTarget(
            name="Cargo Cache",
            path="~/.cargo/registry/cache",
            category="cache",
            description="Rust package cache"
        ),
        
        # Browser Caches (Atmospheric)
        "safari": CleanupTarget(
            name="Safari Cache",
            path="~/Library/Caches/com.apple.Safari",
            category="cache",
            description="Safari browser cache"
        ),
        "chrome": CleanupTarget(
            name="Chrome Cache",
            path="~/Library/Caches/Google/Chrome/Default/Cache",
            category="cache",
            description="Chrome browser cache"
        ),
        
        # System Caches
        "system_cache": CleanupTarget(
            name="User Caches",
            path="~/Library/Caches",
            category="cache",
            safe_to_delete=False,  # Only specific subdirs
            description="General user caches"
        ),
    }
    
    # MoIE Trinity - models to KEEP
    TRINITY_MODELS = [
        "llama3.2",    # Architect
        "mistral",     # Executioner
        "phi",         # Critic
    ]
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.results: List[CleanupResult] = []
        self.total_freed = 0
    
    def get_directory_size(self, path: str) -> int:
        """Get size of directory in bytes"""
        total = 0
        path = Path(path).expanduser()
        
        if not path.exists():
            return 0
        
        try:
            if path.is_file():
                return path.stat().st_size
            
            for entry in path.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except:
                        pass
        except:
            pass
        
        return total
    
    def format_size(self, bytes_count: int) -> str:
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
    
    def scan_targets(self) -> List[CleanupTarget]:
        """Scan and measure all cleanup targets"""
        print("🔍 Scanning for entropy...\n")
        
        targets = []
        for key, target in self.DEFAULT_TARGETS.items():
            path = Path(target.path).expanduser()
            size = self.get_directory_size(str(path))
            
            target.size_bytes = size
            targets.append(target)
            
            if size > 0:
                print(f"   {target.name}: {self.format_size(size)}")
        
        return targets
    
    def clean_target(self, target: CleanupTarget) -> CleanupResult:
        """Clean a single target"""
        path = Path(target.path).expanduser()
        
        if not path.exists():
            return CleanupResult(
                target=target.name,
                freed_bytes=0,
                success=True,
                message="Already clean"
            )
        
        if not target.safe_to_delete:
            return CleanupResult(
                target=target.name,
                freed_bytes=0,
                success=False,
                message="Marked as unsafe - manual review required"
            )
        
        size = target.size_bytes
        
        if self.dry_run:
            return CleanupResult(
                target=target.name,
                freed_bytes=size,
                success=True,
                message=f"Would free {self.format_size(size)} (dry run)"
            )
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            
            return CleanupResult(
                target=target.name,
                freed_bytes=size,
                success=True,
                message=f"Freed {self.format_size(size)}"
            )
        except Exception as e:
            return CleanupResult(
                target=target.name,
                freed_bytes=0,
                success=False,
                message=f"Error: {e}"
            )
    
    def prune_models(self) -> List[str]:
        """Prune Ollama models not in Trinity"""
        print("\n🤖 Auditing Ollama models...")
        
        pruned = []
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print("   ⚠️ Could not list Ollama models")
                return []
            
            # Parse model list
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                
                model_name = parts[0]
                model_base = model_name.split(':')[0]
                
                # Check if in Trinity
                in_trinity = any(t in model_base for t in self.TRINITY_MODELS)
                
                if in_trinity:
                    print(f"   ✅ KEEP: {model_name} (Trinity)")
                else:
                    print(f"   🗑️  PRUNE: {model_name} (Not in Trinity)")
                    
                    if not self.dry_run:
                        try:
                            subprocess.run(
                                ["ollama", "rm", model_name],
                                capture_output=True,
                                timeout=60
                            )
                        except:
                            pass
                    
                    pruned.append(model_name)
            
        except Exception as e:
            print(f"   ⚠️ Error auditing models: {e}")
        
        return pruned
    
    def run_package_cleanup(self):
        """Run package manager cleanup commands"""
        print("\n📦 Running package manager cleanup...")
        
        commands = [
            ("brew cleanup -s", "Homebrew"),
            ("npm cache clean --force", "NPM"),
            ("pip cache purge", "Pip"),
        ]
        
        for cmd, name in commands:
            if self.dry_run:
                print(f"   Would run: {cmd}")
            else:
                try:
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
                    print(f"   ✅ {name} cleaned")
                except:
                    print(f"   ⚠️ {name} cleanup failed")
    
    def clean(self, include_models: bool = False) -> CleanupReport:
        """Run full cleanup"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🧹 ENTROPY JANITOR                                          ║
║  Sovereign Pruning Protocol                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        if self.dry_run:
            print("⚠️  DRY RUN MODE - No files will be deleted\n")
        
        # Scan targets
        targets = self.scan_targets()
        
        # Clean targets
        print("\n🗑️  Cleaning targets...")
        for target in targets:
            if target.size_bytes > 0 and target.safe_to_delete:
                result = self.clean_target(target)
                self.results.append(result)
                self.total_freed += result.freed_bytes
                
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.target}: {result.message}")
        
        # Package cleanup
        self.run_package_cleanup()
        
        # Model pruning
        pruned_models = []
        if include_models:
            pruned_models = self.prune_models()
        
        # Generate report
        report = CleanupReport(
            timestamp=datetime.now().isoformat(),
            total_freed_bytes=self.total_freed,
            targets_cleaned=sum(1 for r in self.results if r.success),
            targets_failed=sum(1 for r in self.results if not r.success),
            results=self.results,
            models_pruned=pruned_models
        )
        
        self.print_report(report)
        return report
    
    def print_report(self, report: CleanupReport):
        """Print cleanup report"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 ENTROPY REDUCTION REPORT                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Total Freed: {self.format_size(report.total_freed_bytes):<45} ║
║  Targets Cleaned: {report.targets_cleaned:<42} ║
║  Targets Failed: {report.targets_failed:<43} ║
║  Models Pruned: {len(report.models_pruned):<44} ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        if self.dry_run:
            print("💡 Run with --execute to actually clean these targets")


# ============================================================================
# 🧪 CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entropy Janitor - Sovereign Pruning Protocol")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry run)")
    parser.add_argument("--models", action="store_true", help="Include Ollama model pruning")
    
    args = parser.parse_args()
    
    janitor = EntropyJanitor(dry_run=not args.execute)
    janitor.clean(include_models=args.models)

#!/usr/bin/env python3
"""
🎮 SOVEREIGN CLI
================
Simple command-line interface for the Universal Controller.

Usage:
    python cli.py                    # Interactive REPL
    python cli.py query "your query" # Single query
    python cli.py status             # Show module status
    python cli.py dashboard          # Open web dashboard
"""

import sys
import webbrowser
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from controller import create_sovereign


def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🎮  SOVEREIGN CLI                                       ║
    ║   ───────────────────────────────────────────────────────║
    ║   Universal Controller • Snap-In Architecture            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def cmd_status(ctrl):
    """Show status of all modules"""
    print(ctrl.status())


def cmd_query(ctrl, query: str):
    """Process a single query"""
    result = ctrl.process(query)
    
    if result["axiom_pre"]:
        pre = result["axiom_pre"]
        print(f"\n⚖️  Axiom Pre-Check: λ={pre['λ']:.2f} α={pre['α']:.2f} σ={pre['σ']:.2f} γ={pre['γ']:.2f}")
    
    print(f"\n📝 Response:\n{result['response']}")
    
    if result["axiom_post"] and not result["blocked"]:
        post = result["axiom_post"]
        print(f"\n⚖️  Axiom Post-Check: λ={post['λ']:.2f} α={post['α']:.2f} σ={post['σ']:.2f} γ={post['γ']:.2f}")
        print(f"   Alignment Score: {post['total']:.2f}/4.5 {'✅' if post['passed'] else '❌'}")


def cmd_dashboard():
    """Open the web dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        webbrowser.open(f"file://{dashboard_path}")
        print("🌐 Opening dashboard in browser...")
    else:
        print("❌ Dashboard not found. Run from sovereign_universal directory.")


def cmd_repl(ctrl):
    """Interactive REPL"""
    ctrl.repl()


def main():
    args = sys.argv[1:]
    
    if not args:
        # Default: REPL mode
        print_banner()
        ctrl = create_sovereign()
        cmd_repl(ctrl)
        return
    
    command = args[0].lower()
    
    if command == "dashboard":
        cmd_dashboard()
    
    elif command == "status":
        print_banner()
        ctrl = create_sovereign()
        cmd_status(ctrl)
    
    elif command == "query" and len(args) > 1:
        print_banner()
        ctrl = create_sovereign()
        query = " ".join(args[1:])
        cmd_query(ctrl, query)
    
    elif command == "help":
        print(__doc__)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'python cli.py help' for usage information.")


if __name__ == "__main__":
    main()

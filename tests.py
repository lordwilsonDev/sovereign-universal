#!/usr/bin/env python3
"""
🧪 SOVEREIGN UNIVERSAL - TEST SUITE
====================================
20 tests to verify all components and identify gaps.
"""

import sys
import time
from typing import Tuple

# Results tracking
results = []

def test(name: str, func) -> Tuple[bool, str]:
    """Run a test and record result"""
    try:
        result = func()
        if result is True or (isinstance(result, tuple) and result[0]):
            msg = result[1] if isinstance(result, tuple) else "✓"
            results.append((name, True, msg))
            print(f"  ✅ {name}")
            return True, msg
        else:
            msg = result[1] if isinstance(result, tuple) else "Failed"
            results.append((name, False, msg))
            print(f"  ❌ {name}: {msg}")
            return False, msg
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  ❌ {name}: {e}")
        return False, str(e)

print("🧪 SOVEREIGN UNIVERSAL TEST SUITE")
print("=" * 60)
print()

# ============================================================================
# TEST 1-5: Core Controller
# ============================================================================
print("📦 CORE CONTROLLER")
print("-" * 40)

def test_1_import():
    from controller import SovereignController
    return True

def test_2_create_controller():
    from controller import create_sovereign
    ctrl = create_sovereign()
    return ctrl is not None

def test_3_modules_loaded():
    from controller import create_sovereign
    ctrl = create_sovereign()
    expected = ["axiom_bridge", "vector_memory", "ollama_llm"]
    loaded = list(ctrl.modules.keys())
    missing = [m for m in expected if m not in loaded]
    if missing:
        return False, f"Missing: {missing}"
    return True

def test_4_status():
    from controller import create_sovereign
    ctrl = create_sovereign()
    status = ctrl.status()
    return "MODULE STATUS" in status

def test_5_enable_tools():
    from controller import create_sovereign
    ctrl = create_sovereign()
    tools = ctrl.enable_tools()
    return len(tools.tools) >= 5, f"{len(tools.tools)} tools"

test("1. Import controller", test_1_import)
test("2. Create controller", test_2_create_controller)
test("3. Modules loaded", test_3_modules_loaded)
test("4. Status works", test_4_status)
test("5. Enable tools", test_5_enable_tools)

# ============================================================================
# TEST 6-9: Axiom Module
# ============================================================================
print()
print("⚖️ AXIOM MODULE")
print("-" * 40)

def test_6_axiom_safe():
    from controller import AxiomModule
    axiom = AxiomModule()
    axiom.snap_in()
    score = axiom.process("Help me learn something new")
    return score.passed, f"Score: {score.weighted_total:.2f}"

def test_7_axiom_block():
    from controller import AxiomModule
    axiom = AxiomModule()
    axiom.snap_in()
    score = axiom.process("rm -rf / delete everything")
    return not score.passed, f"Blocked: σ={score.safety}"

def test_8_axiom_weights():
    from controller import AxiomModule
    axiom = AxiomModule()
    axiom.snap_in()
    score = axiom.process("test")
    expected = score.love + score.abundance + 1.5*score.safety + score.growth
    return abs(score.weighted_total - expected) < 0.01

def test_9_axiom_veto():
    from controller import AxiomModule
    axiom = AxiomModule()
    axiom.snap_in()
    # Safety = 0 should block regardless of other scores
    score = axiom.process("sudo hack exploit")
    return score.safety == 0.0 and not score.passed

test("6. Axiom passes safe input", test_6_axiom_safe)
test("7. Axiom blocks dangerous input", test_7_axiom_block)
test("8. Axiom weights correct (σ=1.5x)", test_8_axiom_weights)
test("9. Safety veto works", test_9_axiom_veto)

# ============================================================================
# TEST 10-12: Memory Module
# ============================================================================
print()
print("💾 MEMORY MODULE")
print("-" * 40)

def test_10_memory_init():
    from controller import MemoryModule
    mem = MemoryModule(storage_path="/tmp/test_sovereign_memory")
    mem.snap_in()
    return True

def test_11_memory_store():
    from controller import MemoryModule
    mem = MemoryModule(storage_path="/tmp/test_sovereign_memory")
    mem.snap_in()
    mid = mem.process({"action": "store", "content": "Test memory content"})
    return mid is not None and mid.startswith("mem_")

def test_12_memory_search():
    from controller import MemoryModule
    mem = MemoryModule(storage_path="/tmp/test_sovereign_memory")
    mem.snap_in()
    results = mem.process({"action": "search", "query": "test"})
    return isinstance(results, list)

test("10. Memory init", test_10_memory_init)
test("11. Memory store", test_11_memory_store)
test("12. Memory search", test_12_memory_search)

# ============================================================================
# TEST 13-15: Tool Registry
# ============================================================================
print()
print("🔧 TOOL REGISTRY")
print("-" * 40)

def test_13_tool_registry_init():
    from modules.tool_registry import ToolRegistry
    reg = ToolRegistry()
    return True

def test_14_tool_register():
    from modules.tool_registry import ToolRegistry
    reg = ToolRegistry()
    reg.register("test_tool", lambda x: x, "Test", {"x": "any"})
    return "test_tool" in reg.tools

def test_15_tool_parse():
    from modules.tool_registry import ToolRegistry
    reg = ToolRegistry()
    text = '<tool>get_time({})</tool>'
    calls = reg.parse_tool_calls(text)
    return len(calls) == 1 and calls[0].tool_name == "get_time"

test("13. Tool registry init", test_13_tool_registry_init)
test("14. Tool register", test_14_tool_register)
test("15. Tool parse from LLM output", test_15_tool_parse)

# ============================================================================
# TEST 16-18: Integration
# ============================================================================
print()
print("🔗 INTEGRATION")
print("-" * 40)

def test_16_full_pipeline():
    from controller import create_sovereign
    ctrl = create_sovereign()
    result = ctrl.process("What is 2+2?")
    return result["response"] is not None and not result["blocked"]

def test_17_pipeline_axiom_check():
    from controller import create_sovereign
    ctrl = create_sovereign()
    result = ctrl.process("Hello world")
    return result["axiom_pre"] is not None and result["axiom_post"] is not None

def test_18_tools_in_pipeline():
    from controller import create_sovereign
    ctrl = create_sovereign()
    ctrl.enable_tools()
    # Check that tools prompt is added
    return ctrl.tools is not None and len(ctrl.tools.tools) >= 5

test("16. Full pipeline (query → response)", test_16_full_pipeline)
test("17. Pipeline has axiom checks", test_17_pipeline_axiom_check)
test("18. Tools integrate with pipeline", test_18_tools_in_pipeline)

# ============================================================================
# TEST 19-20: DSPy Module
# ============================================================================
print()
print("🧠 DSPY MODULE")
print("-" * 40)

def test_19_dspy_import():
    try:
        from modules.dspy_axiom import configure_dspy_ollama, AxiomInverter
        return True
    except ImportError as e:
        return False, str(e)

def test_20_dspy_configure():
    try:
        from modules.dspy_axiom import configure_dspy_ollama
        lm = configure_dspy_ollama()
        return lm is not None
    except Exception as e:
        return False, str(e)

test("19. DSPy import", test_19_dspy_import)
test("20. DSPy configure", test_20_dspy_configure)

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)

print(f"  ✅ Passed: {passed}")
print(f"  ❌ Failed: {failed}")
print(f"  📈 Score:  {passed}/{len(results)} ({100*passed/len(results):.0f}%)")

if failed > 0:
    print()
    print("❌ FAILED TESTS:")
    for name, passed, msg in results:
        if not passed:
            print(f"  - {name}: {msg}")

print()
print("🔍 RECOMMENDATIONS:")
recommendations = []

# Check for common issues
if any("DSPy" in name and not p for name, p, _ in results):
    recommendations.append("- Install dspy-ai: pip install dspy-ai")

if any("Memory" in name and not p for name, p, _ in results):
    recommendations.append("- Check Ollama is running for embeddings")

if any("Ollama" in name and not p for name, p, _ in results):
    recommendations.append("- Ensure Ollama is running: ollama serve")

if not recommendations:
    recommendations.append("- All core systems working! 🎉")
    recommendations.append("- Consider adding more tools")
    recommendations.append("- Consider adding async support")
    recommendations.append("- Consider adding rate limiting")

for r in recommendations:
    print(f"  {r}")

print()

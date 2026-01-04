#!/usr/bin/env python3
"""
🔧 SOVEREIGN TOOL REGISTRY
===========================
Custom tool calling system with Axiom verification.
Register tools, LLM decides when to call them, all calls are verified.

Usage:
    registry = ToolRegistry()
    registry.register("search", search_fn, "Search the web")
    result = registry.execute("search", {"query": "AI safety"})
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
import json
import re


@dataclass
class Tool:
    """A registered tool"""
    name: str
    function: Callable
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    requires_axiom_check: bool = True
    
    def to_prompt(self) -> str:
        """Format tool for LLM prompt"""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({params}): {self.description}"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass  
class ToolCall:
    """A parsed tool call from LLM output"""
    tool_name: str
    arguments: Dict[str, Any]
    raw: str = ""


@dataclass
class ToolResult:
    """Result of a tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    axiom_blocked: bool = False


class ToolRegistry:
    """
    Custom tool registry for Sovereign Controller.
    
    Features:
    - Register/unregister tools dynamically
    - Parse tool calls from LLM output
    - Execute with axiom verification
    - Built-in tools for memory, search, code execution
    - Recursion detection and execution depth limits
    """
    
    # Maximum execution depth to prevent infinite recursion
    MAX_EXECUTION_DEPTH = 10
    
    # Pattern to match tool calls: <tool>name({"arg": "value"})</tool>
    TOOL_PATTERN = re.compile(
        r'<tool>(\w+)\((\{.*?\})\)</tool>',
        re.DOTALL
    )
    
    # Alternative pattern: [TOOL: name(args)]
    ALT_PATTERN = re.compile(
        r'\[TOOL:\s*(\w+)\((.*?)\)\]',
        re.DOTALL
    )
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._axiom_checker = None
        self._execution_depth = 0  # Track current execution depth
        self._execution_stack = []  # Track call stack for recursion detection
    
    def set_axiom_checker(self, checker: Callable[[str], bool]):
        """Set function to verify axiom compliance"""
        self._axiom_checker = checker
    
    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, str] = None,
        requires_axiom_check: bool = True
    ) -> None:
        """Register a new tool"""
        self.tools[name] = Tool(
            name=name,
            function=function,
            description=description,
            parameters=parameters or {},
            requires_axiom_check=requires_axiom_check
        )
        print(f"🔧 Registered tool: {name}")
    
    def unregister(self, name: str) -> bool:
        """Remove a tool"""
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def list_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tools_prompt(self) -> str:
        """Generate prompt section describing available tools"""
        if not self.tools:
            return ""
        
        lines = [
            "\n## Available Tools",
            "You can call tools using: <tool>name({\"arg\": \"value\"})</tool>",
            ""
        ]
        for tool in self.tools.values():
            lines.append(tool.to_prompt())
        
        return "\n".join(lines)
    
    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from LLM output"""
        calls = []
        
        # Try primary pattern
        for match in self.TOOL_PATTERN.finditer(text):
            try:
                name = match.group(1)
                args_str = match.group(2)
                args = json.loads(args_str)
                calls.append(ToolCall(
                    tool_name=name,
                    arguments=args,
                    raw=match.group(0)
                ))
            except json.JSONDecodeError:
                continue
        
        # Try alternative pattern
        for match in self.ALT_PATTERN.finditer(text):
            try:
                name = match.group(1)
                args_str = match.group(2).strip()
                # Try to parse as JSON, or as simple key=value
                if args_str.startswith("{"):
                    args = json.loads(args_str)
                else:
                    args = {"input": args_str}
                calls.append(ToolCall(
                    tool_name=name,
                    arguments=args,
                    raw=match.group(0)
                ))
            except:
                continue
        
        return calls
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool with axiom verification and recursion detection"""
        
        # Check for recursion/depth limit
        if self._execution_depth >= self.MAX_EXECUTION_DEPTH:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Maximum execution depth ({self.MAX_EXECUTION_DEPTH}) exceeded - possible recursion"
            )
        
        # Check for direct recursion
        call_sig = f"{tool_name}:{hash(frozenset(arguments.items()) if arguments else frozenset())}"
        if call_sig in self._execution_stack:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="Recursive tool call detected - blocked"
            )
        
        # Check tool exists
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        
        # Axiom check on arguments
        if tool.requires_axiom_check and self._axiom_checker:
            args_str = json.dumps(arguments)
            if not self._axiom_checker(args_str):
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error="Blocked by Axiom Safety",
                    axiom_blocked=True
                )
        
        # Execute with depth tracking
        self._execution_depth += 1
        self._execution_stack.append(call_sig)
        
        try:
            result = tool.function(**arguments)
            
            # Axiom check on result
            if tool.requires_axiom_check and self._axiom_checker:
                result_str = str(result)
                if not self._axiom_checker(result_str):
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        result=None,
                        error="Result blocked by Axiom Safety",
                        axiom_blocked=True
                    )
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )
        finally:
            # Always reset depth tracking
            self._execution_depth -= 1
            if call_sig in self._execution_stack:
                self._execution_stack.remove(call_sig)
    
    def execute_all(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls"""
        return [self.execute(c.tool_name, c.arguments) for c in calls]


# ============================================================================
# 🔧 BUILT-IN TOOLS
# ============================================================================

def create_builtin_tools(controller) -> ToolRegistry:
    """Create registry with built-in tools for Sovereign Controller"""
    registry = ToolRegistry()
    
    # Set axiom checker from controller
    axiom_mod = controller.get("axiom_bridge")
    if axiom_mod:
        def check_axiom(text: str) -> bool:
            score = axiom_mod.process(text)
            return score.passed
        registry.set_axiom_checker(check_axiom)
    
    # === MEMORY TOOLS ===
    
    def search_memory(query: str, limit: int = 5) -> str:
        """Search vector memory"""
        mem = controller.get("vector_memory")
        if not mem:
            return "Memory module not available"
        results = mem.process({"action": "search", "query": query})
        if not results:
            return "No memories found"
        return "\n".join([f"- {m.content[:200]}" for m in results[:limit]])
    
    registry.register(
        "search_memory",
        search_memory,
        "Search stored memories for relevant context",
        {"query": "string", "limit": "int (optional)"}
    )
    
    def store_memory(content: str) -> str:
        """Store new memory"""
        mem = controller.get("vector_memory")
        if not mem:
            return "Memory module not available"
        mid = mem.process({"action": "store", "content": content})
        return f"Stored memory: {mid}"
    
    registry.register(
        "store_memory",
        store_memory,
        "Store information in long-term memory",
        {"content": "string"}
    )
    
    # === AXIOM TOOLS ===
    
    def check_alignment(text: str) -> str:
        """Check axiom alignment of text"""
        axiom = controller.get("axiom_bridge")
        if not axiom:
            return "Axiom module not available"
        score = axiom.process(text)
        return f"λ={score.love:.2f} α={score.abundance:.2f} σ={score.safety:.2f} γ={score.growth:.2f} | Passed: {score.passed}"
    
    registry.register(
        "check_alignment",
        check_alignment,
        "Check if text aligns with the Four Axioms",
        {"text": "string"},
        requires_axiom_check=False  # This IS the axiom check
    )
    
    # === SYSTEM TOOLS ===
    
    def get_time() -> str:
        """Get current time"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    registry.register(
        "get_time",
        get_time,
        "Get the current date and time",
        {},
        requires_axiom_check=False
    )
    
    def calculate(expression: str) -> str:
        """Safe math evaluation using AST - NO EVAL"""
        import ast
        import operator
        
        # Allowed operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Invalid constant: {node.value}")
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](left, right)
                raise ValueError(f"Unsupported operator: {op_type}")
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand>
                operand = safe_eval(node.operand)
                op_type = type(node.op)
                if op_type in ops:
                    return ops[op_type](operand)
                raise ValueError(f"Unsupported unary operator: {op_type}")
            elif isinstance(node, ast.Expression):
                return safe_eval(node.body)
            else:
                raise ValueError(f"Unsupported: {type(node)}")
        
        try:
            # Limit expression length
            if len(expression) > 1000:
                return "Error: Expression too long"
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    registry.register(
        "calculate",
        calculate,
        "Perform mathematical calculations (safe, no code execution)",
        {"expression": "string (e.g. '2 + 2 * 3')"},
        requires_axiom_check=False
    )
    
    return registry


# ============================================================================
# 🧪 TEST
# ============================================================================

if __name__ == "__main__":
    print("🔧 Tool Registry Test\n")
    
    registry = ToolRegistry()
    
    # Register test tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    registry.register("greet", greet, "Greet someone", {"name": "string"})
    
    # Test parsing
    llm_output = """
    I'll help you with that. Let me greet you.
    <tool>greet({"name": "World"})</tool>
    """
    
    calls = registry.parse_tool_calls(llm_output)
    print(f"Parsed {len(calls)} tool calls:")
    for call in calls:
        print(f"  - {call.tool_name}({call.arguments})")
    
    # Execute
    for call in calls:
        result = registry.execute(call.tool_name, call.arguments)
        print(f"\nResult: {result}")

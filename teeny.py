# tinychain.py - Enhanced minimalist LLM agent framework with transparency
import json
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic, Set, Tuple
import inspect
import re
from datetime import datetime
import hashlib
from collections import defaultdict
import time

# For visualization support - allow graceful degradation if not available
try:
    import matplotlib.pyplot as plt
    import io
    import base64
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

T = TypeVar('T')

# =========== Core Primitives ===========

class Prompt:
    """Minimal prompt template system."""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the prompt template with provided values."""
        return self.template.format(**kwargs)


class ToolCache:
    """Cache for tool calls to avoid redundant executions."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.call_counts = defaultdict(int)
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate a unique cache key for a tool call."""
        # Sort the args dictionary to ensure consistent keys
        sorted_args = json.dumps(args, sort_keys=True)
        key_string = f"{tool_name}:{sorted_args}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result for a tool call, if available."""
        key = self._generate_key(tool_name, args)
        result = self.cache.get(key)
        
        if result:
            self.call_counts[key] += 1
            self.hits += 1
            return result
        
        self.misses += 1
        return None
    
    def set(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store a tool call result in the cache."""
        key = self._generate_key(tool_name, args)
        self.cache[key] = result
        self.call_counts[key] = 1
        
        # Evict least used entries if cache exceeds max size
        if len(self.cache) > self.max_size:
            least_used_key = min(self.call_counts.items(), key=lambda x: x[1])[0]
            del self.cache[least_used_key]
            del self.call_counts[least_used_key]
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.call_counts.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "most_used_tools": sorted(
                [(tool, count) for tool, count in self.call_counts.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }


class TaskTracker:
    """Tracks the status of subtasks in a complex query."""
    
    def __init__(self):
        self.tasks = []
        self.completed_tasks = set()
        self.current_task_index = 0
    
    def set_tasks(self, tasks: List[str]) -> None:
        """Set the list of tasks to be completed."""
        self.tasks = tasks
        self.completed_tasks = set()
        self.current_task_index = 0
    
    def mark_completed(self, task_idx: int) -> None:
        """Mark a task as completed."""
        if 0 <= task_idx < len(self.tasks):
            self.completed_tasks.add(task_idx)
            # Update current task index to the next uncompleted task
            for i in range(len(self.tasks)):
                if i not in self.completed_tasks:
                    self.current_task_index = i
                    break
            else:
                # All tasks completed
                self.current_task_index = len(self.tasks)
    
    def get_next_incomplete_task(self) -> Optional[Tuple[int, str]]:
        """Get the next task that needs to be completed."""
        for idx, task in enumerate(self.tasks):
            if idx not in self.completed_tasks:
                return (idx, task)
        return None
    
    def all_tasks_completed(self) -> bool:
        """Check if all tasks have been completed."""
        return len(self.completed_tasks) == len(self.tasks)
    
    def get_completion_status(self) -> Dict[str, Any]:
        """Get the status of all tasks."""
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": [self.tasks[i] for i in range(len(self.tasks)) if i not in self.completed_tasks],
            "all_completed": self.all_tasks_completed()
        }


class Tool:
    """Represents a callable tool for the LLM to use."""
    
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.signature = self._get_signature()
        self.call_count = 0
        self.cumulative_execution_time = 0
        
    def _get_signature(self) -> Dict[str, Any]:
        """Extract parameter information from the function signature."""
        sig = inspect.signature(self.func)
        params = {}
        
        for param_name, param in sig.parameters.items():
            # Skip self for methods
            if param_name == 'self':
                continue
                
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else "string"
            if hasattr(param_type, "__name__"):
                type_name = param_type.__name__
            else:
                type_name = str(param_type).replace("<class '", "").replace("'>", "")
                
            params[param_name] = {
                "type": type_name,
                "required": param.default == inspect.Parameter.empty
            }
            
        return params
    
    def __call__(self, *args, **kwargs):
        """Execute the tool with the provided arguments."""
        self.call_count += 1
        
        start_time = time.time()
        try:
            # Check if the function returns visualizations
            result = self.func(*args, **kwargs)
            
            # For functions that return a dict with visualization data
            if isinstance(result, dict) and "visualization" in result:
                # The function already returns a properly formatted result
                if "status" in result:
                    return result
                
                # Wrap the result in our standard format
                return {
                    "status": "success",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check if matplotlib figures exist and this isn't already a dict
            if VISUALIZATION_AVAILABLE and plt.get_fignums() and not isinstance(result, dict):
                # Capture the current figure
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to base64
                img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()  # Close the figure to avoid memory leaks
                
                # Return the result along with the visualization
                return {
                    "status": "success",
                    "result": result,
                    "visualization": {
                        "type": "image/png",
                        "data": img_data
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # Standard result with no visualization
            return {
                "status": "success",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            execution_time = time.time() - start_time
            self.cumulative_execution_time += execution_time


class Memory:
    """Enhanced memory store with better tracking of conversation, reasoning, and persistent state."""
    
    def __init__(self):
        self.storage = {}
        self.history = []
        self.tool_calls = []
        self.reasoning_steps = []
        self.reflection_steps = []
        self.session_state = {}  # For maintaining state between queries
        self.seen_tool_calls = set()  # Track tool calls to detect duplicates
    
    def set(self, key: str, value: Any) -> None:
        """Store a value by key."""
        self.storage[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return self.storage.get(key, default)
    
    def set_session_state(self, key: str, value: Any) -> None:
        """Store a value in the session state that persists between queries."""
        self.session_state[key] = value
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the session state."""
        return self.session_state.get(key, default)
    
    def clear_session_state(self) -> None:
        """Clear the session state."""
        self.session_state.clear()
    
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the conversation history."""
        # Add a timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
            
        self.history.append(entry)
        
        # Track specialized entries
        if entry.get("type") == "reasoning":
            self.reasoning_steps.append(entry)
        elif entry.get("type") == "reflection":
            self.reflection_steps.append(entry)
        elif entry.get("type") == "tool_call":
            self.tool_calls.append(entry)
            
            # Track this tool call to detect duplicates
            tool_name = entry.get("tool", "unknown")
            args = json.dumps(entry.get("args", {}), sort_keys=True)
            call_signature = f"{tool_name}:{args}"
            self.seen_tool_calls.add(call_signature)
    
    def is_duplicate_tool_call(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if a tool call with the same parameters has already been made."""
        args_str = json.dumps(args, sort_keys=True)
        call_signature = f"{tool_name}:{args_str}"
        return call_signature in self.seen_tool_calls
    
    def get_history(self, last_n: Optional[int] = None, 
                   entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get conversation history, optionally filtered by entry types and limited to last n entries."""
        if entry_types:
            filtered_history = [entry for entry in self.history if entry.get("type") in entry_types]
        else:
            filtered_history = self.history
            
        if last_n is not None:
            return filtered_history[-last_n:]
        return filtered_history
    
    def get_unique_tool_results(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get unique results from tool calls, optionally filtered by tool name."""
        seen_results = set()
        unique_results = []
        
        for entry in self.history:
            if entry.get("role") == "system" and "Tool result:" in entry.get("content", ""):
                if tool_name is None or tool_name in entry.get("content", ""):
                    result_content = entry.get("content", "").replace("Tool result:", "").strip()
                    
                    # Only add if we haven't seen this result before
                    if result_content not in seen_results:
                        seen_results.add(result_content)
                        unique_results.append(result_content)
        
        return unique_results
    
    def get_last_reasoning(self) -> Optional[str]:
        """Get the most recent reasoning step."""
        for entry in reversed(self.history):
            if entry.get("type") == "reasoning":
                return entry.get("content", "").replace("Reasoning:", "").strip()
        return None
    
    def get_last_reflection(self) -> Optional[str]:
        """Get the most recent reflection step."""
        for entry in reversed(self.history):
            if entry.get("type") == "reflection":
                return entry.get("content", "").replace("Reflection:", "").strip()
        return None
    
    def get_reasoning_chain(self) -> List[str]:
        """Get the full chain of reasoning steps."""
        return [entry.get("content", "").replace("Reasoning:", "").strip() 
                for entry in self.history if entry.get("type") == "reasoning"]
    
    def get_reflection_chain(self) -> List[str]:
        """Get the full chain of reflection steps."""
        return [entry.get("content", "").replace("Reflection:", "").strip() 
                for entry in self.history if entry.get("type") == "reflection"]
    
    def get_summarized_reasoning(self) -> str:
        """Get a summarized version of all reasoning steps."""
        if not self.reasoning_steps:
            return "No reasoning steps available."
        
        # For simplicity, just concatenate the most recent reasoning
        if len(self.reasoning_steps) > 0:
            return self.reasoning_steps[-1].get("content", "").replace("Reasoning:", "").strip()
        return ""
    
    def save_to_file(self, filename: str) -> None:
        """Save the memory state to a file."""
        with open(filename, "w") as f:
            json.dump({
                "history": self.history,
                "session_state": self.session_state
            }, f, indent=2)
    
    def load_from_file(self, filename: str) -> None:
        """Load the memory state from a file."""
        with open(filename, "r") as f:
            data = json.load(f)
            self.history = data.get("history", [])
            self.session_state = data.get("session_state", {})
            
            # Rebuild specialized entry lists
            self.tool_calls = []
            self.reasoning_steps = []
            self.reflection_steps = []
            self.seen_tool_calls = set()
            
            for entry in self.history:
                if entry.get("type") == "reasoning":
                    self.reasoning_steps.append(entry)
                elif entry.get("type") == "reflection":
                    self.reflection_steps.append(entry)
                elif entry.get("type") == "tool_call":
                    self.tool_calls.append(entry)
                    
                    # Rebuild seen tool calls set
                    tool_name = entry.get("tool", "unknown")
                    args = json.dumps(entry.get("args", {}), sort_keys=True)
                    call_signature = f"{tool_name}:{args}"
                    self.seen_tool_calls.add(call_signature)


class Chain:
    """Composes multiple operations into a single workflow."""
    
    def __init__(self):
        self.steps = []
        self.memory = Memory()
        self.execution_times = []
        self.step_results = []
    
    def add(self, step_func: Callable[[Dict[str, Any], Memory], Dict[str, Any]]) -> 'Chain':
        """Add a step to the chain."""
        self.steps.append(step_func)
        return self
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all steps in the chain sequentially."""
        context = inputs
        self.execution_times = []
        self.step_results = []
        
        for i, step in enumerate(self.steps):
            start_time = time.time()
            
            try:
                # Get step name for reporting
                step_name = step.__name__ if hasattr(step, "__name__") else f"step_{i}"
                
                # Execute the step
                step_result = step(context, self.memory)
                
                # Record execution info
                execution_time = time.time() - start_time
                self.execution_times.append((step_name, execution_time))
                self.step_results.append((step_name, step_result))
                
                # Update context with step result
                context.update(step_result)
                
                # Record step execution in memory
                self.memory.add_to_history({
                    "type": "chain_step",
                    "step": step_name,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Error in step {i}: {str(e)}"
                
                # Record error in memory
                self.memory.add_to_history({
                    "type": "chain_error",
                    "step": step_name if 'step_name' in locals() else f"step_{i}",
                    "error": str(e),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Re-raise the exception
                raise e
            
        # Add execution summary to the context
        context["_execution_summary"] = {
            "total_steps": len(self.steps),
            "total_time": sum(time for _, time in self.execution_times),
            "step_times": self.execution_times
        }
        
        return context


class TokenCounter:
    """Track token usage for API calls."""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_cost = 0.0
        self.price_per_1k_prompt = 0.0  # Set based on model
        self.price_per_1k_completion = 0.0  # Set based on model
    
    def update(self, prompt_tokens: int, completion_tokens: int):
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.estimated_cost = (
            (self.prompt_tokens / 1000) * self.price_per_1k_prompt + 
            (self.completion_tokens / 1000) * self.price_per_1k_completion
        )
    
    def set_price(self, model_name: str):
        """Set price based on model."""
        # Default prices for common models
        if "gpt-4" in model_name.lower():
            if "32k" in model_name.lower():
                self.price_per_1k_prompt = 0.06
                self.price_per_1k_completion = 0.12
            else:
                self.price_per_1k_prompt = 0.03
                self.price_per_1k_completion = 0.06
        elif "gpt-3.5-turbo" in model_name.lower():
            self.price_per_1k_prompt = 0.0015
            self.price_per_1k_completion = 0.002
        elif "claude" in model_name.lower():
            if "opus" in model_name.lower():
                self.price_per_1k_prompt = 0.015
                self.price_per_1k_completion = 0.075
            elif "sonnet" in model_name.lower():
                self.price_per_1k_prompt = 0.003
                self.price_per_1k_completion = 0.015
            elif "haiku" in model_name.lower():
                self.price_per_1k_prompt = 0.00025
                self.price_per_1k_completion = 0.00125
        elif "llama" in model_name.lower():
            # DeepInfra prices are estimates
            self.price_per_1k_prompt = 0.0007
            self.price_per_1k_completion = 0.0007
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 6)
        }


class Trace:
    """Captures and formats the execution trace for better explainability."""
    
    def __init__(self, memory: Memory):
        self.memory = memory
    
    def get_full_trace(self) -> List[Dict[str, Any]]:
        """Get the complete execution trace with all steps."""
        return self.memory.history
    
    def get_reasoning_trace(self) -> List[str]:
        """Get only the reasoning steps."""
        return self.memory.get_reasoning_chain()
    
    def get_tool_trace(self) -> List[Dict[str, Any]]:
        """Get only the tool calls and results."""
        return [entry for entry in self.memory.history 
                if entry.get("type") == "tool_call" or 
                (entry.get("role") == "system" and "Tool result:" in entry.get("content", ""))]
    
    def format_trace(self, format_type: str = "text") -> str:
        """Format the trace for display in different formats."""
        if format_type == "text":
            return self._format_text_trace()
        elif format_type == "markdown":
            return self._format_markdown_trace()
        elif format_type == "html":
            return self._format_html_trace()
        elif format_type == "json":
            return json.dumps(self.get_full_trace(), indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_text_trace(self) -> str:
        """Format the trace as plain text."""
        lines = []
        for entry in self.memory.history:
            role = entry.get("role", "").upper()
            content = entry.get("content", "")
            entry_type = entry.get("type", "")
            
            if entry_type == "reasoning":
                lines.append(f"🧠 REASONING: {content}")
            elif entry_type == "reflection":
                lines.append(f"💭 REFLECTION: {content}")
            elif entry_type == "tool_call":
                lines.append(f"🛠️ TOOL CALL ({entry.get('tool', 'unknown')}): {content}")
            elif entry_type == "tool_reasoning":
                lines.append(f"🔍 TOOL REASONING: {content}")
            elif "Tool result:" in content:
                lines.append(f"📊 RESULT: {content.replace('Tool result:', '').strip()}")
            else:
                lines.append(f"{role}: {content}")
            
            lines.append("-" * 50)
            
        return "\n".join(lines)
    
    def _format_markdown_trace(self) -> str:
        """Format the trace as markdown."""
        lines = ["# Execution Trace", ""]
        
        current_step = 1
        for entry in self.memory.history:
            role = entry.get("role", "").upper()
            content = entry.get("content", "")
            entry_type = entry.get("type", "")
            
            if entry_type == "reasoning":
                lines.append(f"## Step {current_step}: Reasoning")
                lines.append(f"_{content}_")
                current_step += 1
            elif entry_type == "reflection":
                lines.append(f"## Step {current_step}: Reflection")
                lines.append(f"_{content}_")
                current_step += 1
            elif entry_type == "tool_call":
                tool = entry.get("tool", "unknown")
                lines.append(f"## Step {current_step}: Tool Call - {tool}")
                lines.append(f"**Parameters:** {entry.get('args', {})}")
                current_step += 1
            elif entry_type == "tool_reasoning":
                lines.append("### Tool Selection Reasoning")
                lines.append(f"_{content}_")
            elif "Tool result:" in content:
                lines.append("### Result")
                lines.append(f"```\n{content.replace('Tool result:', '').strip()}\n```")
                
                # Check if there's a visualization to include
                visualization = entry.get("visualization")
                if visualization and "data" in visualization:
                    img_type = visualization.get("type", "image/png")
                    img_data = visualization.get("data", "")
                    lines.append(f"![Visualization](data:{img_type};base64,{img_data})")
            elif role == "USER":
                lines.append(f"## Query")
                lines.append(f"> {content}")
            elif role == "ASSISTANT" and entry_type == "final_response":
                lines.append(f"## Final Answer")
                lines.append(content)
                
            lines.append("")
            
        return "\n".join(lines)
    
    def _format_html_trace(self) -> str:
        """Format the trace as HTML."""
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Execution Trace</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }",
            "        .step { border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }",
            "        .reasoning { background-color: #f5f5ff; }",
            "        .reflection { background-color: #f5fff5; }",
            "        .tool-call { background-color: #fff5f5; }",
            "        .result { background-color: #fffff5; }",
            "        .query { background-color: #f5f5f5; font-style: italic; }",
            "        .final { background-color: #f0f0f0; font-weight: bold; }",
            "        .title { font-weight: bold; }",
            "        .content { margin-top: 5px; }",
            "        .visualization { max-width: 100%; margin-top: 10px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Execution Trace</h1>"
        ]
        
        for entry in self.memory.history:
            role = entry.get("role", "").upper()
            content = entry.get("content", "")
            entry_type = entry.get("type", "")
            
            if entry_type == "reasoning":
                html_lines.append(f'    <div class="step reasoning">')
                html_lines.append(f'        <div class="title">Reasoning:</div>')
                html_lines.append(f'        <div class="content">{content}</div>')
                html_lines.append(f'    </div>')
            elif entry_type == "reflection":
                html_lines.append(f'    <div class="step reflection">')
                html_lines.append(f'        <div class="title">Reflection:</div>')
                html_lines.append(f'        <div class="content">{content}</div>')
                html_lines.append(f'    </div>')
            elif entry_type == "tool_call":
                tool = entry.get("tool", "unknown")
                args = entry.get("args", {})
                html_lines.append(f'    <div class="step tool-call">')
                html_lines.append(f'        <div class="title">Tool Call: {tool}</div>')
                html_lines.append(f'        <div class="content">Parameters: {json.dumps(args)}</div>')
                html_lines.append(f'    </div>')
            elif "Tool result:" in content:
                result = content.replace("Tool result:", "").strip()
                html_lines.append(f'    <div class="step result">')
                html_lines.append(f'        <div class="title">Result:</div>')
                html_lines.append(f'        <div class="content">{result}</div>')
                
                # Include visualization if available
                visualization = entry.get("visualization")
                if visualization and "data" in visualization:
                    img_type = visualization.get("type", "image/png")
                    img_data = visualization.get("data", "")
                    html_lines.append(f'        <img class="visualization" src="data:{img_type};base64,{img_data}" alt="Visualization" />')
                
                html_lines.append(f'    </div>')
            elif role == "USER":
                html_lines.append(f'    <div class="step query">')
                html_lines.append(f'        <div class="title">Query:</div>')
                html_lines.append(f'        <div class="content">{content}</div>')
                html_lines.append(f'    </div>')
            elif role == "ASSISTANT" and entry_type == "final_response":
                html_lines.append(f'    <div class="step final">')
                html_lines.append(f'        <div class="title">Final Answer:</div>')
                html_lines.append(f'        <div class="content">{content}</div>')
                html_lines.append(f'    </div>')
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        return "\n".join(html_lines)


# =========== LLM Interface ===========

class LLM:
    """Base interface for language models."""
    
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def generate_with_tools(self, prompt: str, tools: List[Tool]) -> Dict[str, Any]:
        """Generate a response that might include tool calls."""
        raise NotImplementedError("Subclasses must implement generate_with_tools()")


# =========== Implementations ===========

class OpenAILLM(LLM):
    """Implementation for OpenAI-compatible APIs."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install it with 'pip install openai'")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.token_counter = TokenCounter()
        self.token_counter.set_price(model)
    
    def generate(self, prompt: str) -> str:
        """Generate a text response using the OpenAI API."""
        # Handle both string and message list formats
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        execution_time = time.time() - start_time
        
        # Update token usage
        if hasattr(response, 'usage'):
            self.token_counter.update(
                response.usage.prompt_tokens, 
                response.usage.completion_tokens
            )
        
        return response.choices[0].message.content
    
    def generate_with_tools(self, prompt, tools: List[Tool]) -> Dict[str, Any]:
        """Generate a response that might include tool calls using OpenAI function calling."""
        # Handle both string and message list formats
        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, str):
            # Parse string format into messages
            lines = prompt.split('\n')
            messages = []
            current_role = None
            current_content = []
            
            for line in lines:
                if line.startswith('system:') or line.startswith('user:') or line.startswith('assistant:'):
                    # Save the previous message if there is one
                    if current_role is not None:
                        messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
                        current_content = []
                    
                    # Start a new message
                    parts = line.split(':', 1)
                    current_role = parts[0].strip()
                    if len(parts) > 1:
                        current_content.append(parts[1].strip())
                else:
                    # Continue the current message
                    if current_role is not None:
                        current_content.append(line)
            
            # Add the last message
            if current_role is not None:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            
            # If we couldn't parse any messages, use the entire prompt as a user message
            if not messages:
                messages = [{"role": "user", "content": prompt}]
        else:
            # If it's neither a list nor a string, convert to string and use as user message
            messages = [{"role": "user", "content": str(prompt)}]
        
        try:
            # Try using OpenAI's function calling interface
            # Convert tool definitions to the OpenAI format
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                param: {"type": info["type"].lower() if hasattr(info["type"], "lower") else str(info["type"])}
                                for param, info in tool.signature.items()
                            },
                            "required": [
                                param for param, info in tool.signature.items() 
                                if info.get("required", True)
                            ]
                        }
                    }
                } for tool in tools
            ]
            
            # Make the API call with tool calling
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto"
            )
            execution_time = time.time() - start_time
            
            # Update token usage
            if hasattr(response, 'usage'):
                self.token_counter.update(
                    response.usage.prompt_tokens, 
                    response.usage.completion_tokens
                )
            
            message = response.choices[0].message
            
            # Check if the model wants to call a tool
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                    
                # Find the matching tool
                called_tool = next((tool for tool in tools if tool.name == tool_name), None)
                
                if called_tool:
                    # Capture the model's reasoning before executing the tool
                    reasoning_prompt = [
                        {"role": "system", "content": "Explain your reasoning for choosing this tool and what you expect to learn from it."},
                        {"role": "user", "content": f"You chose to use the {tool_name} tool with these parameters: {json.dumps(arguments)}. Why did you choose this tool and what do you expect to learn?"}
                    ]
                    
                    reasoning = self.generate(reasoning_prompt)
                    
                    # Execute the tool with the provided arguments
                    tool_result = called_tool(**arguments)
                    
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": arguments,
                        "result": tool_result,
                        "reasoning": reasoning,
                        "execution_time": execution_time
                    }
            
            # If no tool call or tool not found, return the text response
            return {
                "type": "text",
                "content": message.content,
                "execution_time": execution_time
            }
            
        except Exception as e:
            # If function calling fails, fall back to the text-based approach
            # This makes our implementation model-agnostic
            return self._text_based_tool_calling(messages, tools)
    
    def _text_based_tool_calling(self, messages, tools):
        """Fall back to text-based tool calling when function calling is not supported."""
        # Add tool descriptions to the system message
        tools_desc = "\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.signature)}"
            for tool in tools
        ])
        
        tool_instructions = f"""
Available Tools:
{tools_desc}

To use a tool, respond in the following format:

USE TOOL: <tool_name>
PARAMETERS: 
{{
  "param1": "value1",
  "param2": "value2"
}}

Only use one of the tools listed above, and only when necessary. If you don't need to use a tool, just respond normally.
"""
        
        # Find the system message, or add one if it doesn't exist
        system_message_found = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                # Append tool descriptions to existing system message
                messages[i]["content"] = msg["content"] + "\n\n" + tool_instructions
                system_message_found = True
                break
        
        if not system_message_found:
            # Insert a new system message at the beginning
            messages.insert(0, {
                "role": "system", 
                "content": f"You are a helpful assistant with access to tools.\n\n{tool_instructions}"
            })
        
        # Make the API call without tool calling
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        execution_time = time.time() - start_time
        
        # Update token usage
        if hasattr(response, 'usage'):
            self.token_counter.update(
                response.usage.prompt_tokens, 
                response.usage.completion_tokens
            )
        
        message_content = response.choices[0].message.content
        
        # Try to parse a tool call from the response
        tool_call = self._parse_tool_call_from_text(message_content)
        
        if tool_call:
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("args", {})
            
            # Find the matching tool
            called_tool = next((tool for tool in tools if tool.name == tool_name), None)
            
            if called_tool:
                # Capture the model's reasoning with a follow-up query
                reasoning_prompt = [
                    {"role": "system", "content": "Explain your reasoning for choosing this tool and what you expect to learn from it."},
                    {"role": "user", "content": f"You chose to use the {tool_name} tool with these parameters: {json.dumps(arguments)}. Why did you choose this tool and what do you expect to learn?"}
                ]
                
                reasoning = self.generate(reasoning_prompt)
                
                # Execute the tool with the provided arguments
                try:
                    tool_result = called_tool(**arguments)
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": arguments,
                        "result": tool_result,
                        "reasoning": reasoning,
                        "execution_time": execution_time
                    }
                except Exception as e:
                    # If tool execution fails, return an error
                    return {
                        "type": "error",
                        "content": f"I tried to use the {tool_name} tool but encountered an error: {str(e)}",
                        "error": str(e),
                        "tool": tool_name,
                        "args": arguments,
                        "execution_time": execution_time
                    }
        
        # If no tool call detected or parsing failed, return the text response
        return {
            "type": "text",
            "content": message_content,
            "execution_time": execution_time
        }
    
    def _parse_tool_call_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to parse a tool call from text response."""
        # Look for the tool call format
        tool_pattern = r'USE TOOL: (\w+)\s+PARAMETERS:\s+({[\s\S]*?})'
        match = re.search(tool_pattern, text, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            
            try:
                args = json.loads(args_str)
                return {
                    "tool": tool_name,
                    "args": args
                }
            except json.JSONDecodeError:
                # Failed to parse JSON
                return None
                
        return None
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return self.token_counter.get_stats()


# For backward compatibility with the pasted code
CustomOpenAILLM = OpenAILLM


# =========== Agent Implementation ===========

class Agent:
    """Enhanced LLM-powered agent with transparency, reasoning, and task tracking capabilities."""
    
    def __init__(self, llm: LLM, system_prompt: Optional[str] = None):
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.tools = []
        self.memory = Memory()
        self.trace = Trace(self.memory)
        self.tool_cache = ToolCache(max_size=100)
        self.task_tracker = TaskTracker()
    
    def add_tool(self, tool: Tool) -> 'Agent':
        """Add a tool for the agent to use."""
        self.tools.append(tool)
        return self
    
    def add_tools(self, tools: List[Tool]) -> 'Agent':
        """Add multiple tools for the agent to use."""
        self.tools.extend(tools)
        return self
    
    def extract_tasks(self, user_query: str) -> List[str]:
        """Extract subtasks from a complex user query."""
        # If the query has multiple questions or requests, try to extract them
        if " and " in user_query.lower() or " then " in user_query.lower() or "; " in user_query.lower():
            tasks = []
            
            # For simple pattern matching, split on common markers
            for splitter in [" and ", " then ", "; "]:
                if splitter in user_query.lower():
                    parts = user_query.split(splitter)
                    for part in parts:
                        # Clean up the part
                        task = part.strip()
                        if task:
                            tasks.append(task)
                    
                    if tasks:
                        # Don't overprocess - once we find a splitter that works, return those tasks
                        return tasks
            
            # If no tasks were extracted with the basic approach, try a more advanced approach
            # Check for common patterns like "1. ..., 2. ..." or "First... Second..."
            task_pattern = r'(?:^|\n)\s*(?:(\d+)[\.|\)]|(?:First|Second|Third|Fourth|Fifth|Next|Finally))\s+([^\n]+)'
            matches = re.findall(task_pattern, user_query, re.IGNORECASE)
            
            if matches:
                for _, task in matches:
                    if task.strip():
                        tasks.append(task.strip())
                
                if tasks:
                    return tasks
        
        # Default to treating the whole query as a single task
        return [user_query]
    
    def is_task_complete(self, task: str, tool_name: str) -> bool:
        """Check if a task is complete based on the tool called."""
        task_lower = task.lower()
        tool_name_lower = tool_name.lower()
        
        # Check for common task completion patterns
        if ("compare" in task_lower or "performance" in task_lower) and "best_performing" in tool_name_lower:
            return True
        elif ("monte carlo" in task_lower or "simulation" in task_lower) and "monte_carlo" in tool_name_lower:
            return True
        elif "price" in task_lower and "stock_price" in tool_name_lower:
            return True
        elif "change" in task_lower and "price_change" in tool_name_lower:
            return True
        
        # Default: not completed
        return False
    
    def run(self, user_query: str, max_steps: int = 5, verbose: bool = False, 
            return_trace: bool = False, persistent_memory: bool = True) -> Union[tuple, Dict[str, Any]]:
        """
        Run the agent to respond to a user query, using tools if necessary.
        
        Args:
            user_query: The user's question or request
            max_steps: Maximum number of reasoning/tool steps to take
            verbose: Whether to print detailed progress information
            return_trace: Whether to return the full execution trace
            persistent_memory: Whether to maintain memory between queries
            
        Returns:
            If return_trace is False: (final_response, history)
            If return_trace is True: {
                "response": final_response, 
                "history": history,
                "trace": formatted trace,
                "token_usage": token usage statistics,
                "tool_usage": tool usage statistics
            }
        """
        # If not using persistent memory, clear the history but keep session state
        if not persistent_memory:
            session_state = self.memory.session_state
            self.memory = Memory()
            self.memory.session_state = session_state
            self.trace = Trace(self.memory)
        
        # Extract tasks from query
        tasks = self.extract_tasks(user_query)
        if len(tasks) > 1:
            self.task_tracker.set_tasks(tasks)
            if verbose:
                print(f"📋 Extracted {len(tasks)} tasks from query:")
                for i, task in enumerate(tasks):
                    print(f"  {i+1}. {task}")
        
        # Initialize conversation with system prompt
        conversation = [{
            "role": "system",
            "content": self.system_prompt
        }]
        
        # Add user query
        conversation.append({
            "role": "user",
            "content": user_query
        })
        
        # Record in memory
        self.memory.add_to_history({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        step = 0
        final_response = ""
        visualizations = []
        start_time = time.time()
        
        while step < max_steps:
            # Get the next task to complete if we're tracking tasks
            current_task = None
            if len(tasks) > 1 and not self.task_tracker.all_tasks_completed():
                task_info = self.task_tracker.get_next_incomplete_task()
                if task_info:
                    task_idx, task_text = task_info
                    current_task = task_text
                    
                    # Add a message to prompt the model to focus on this task
                    conversation.append({
                        "role": "system",
                        "content": f"Now focus on completing this specific task: {task_text}"
                    })
                    
                    if verbose:
                        print(f"🔍 Now focusing on task {task_idx+1}: {task_text}")
            
            # We will hit the max steps if the model keeps choosing to use tools
            if step == max_steps - 1:
                # For the last step, explicitly ask for a final answer
                if len(tasks) > 1 and not self.task_tracker.all_tasks_completed():
                    conversation.append({
                        "role": "user",
                        "content": "Please provide a comprehensive answer that addresses ALL parts of my original query. Make sure to include information about ALL tasks I asked about."
                    })
                else:
                    conversation.append({
                        "role": "user",
                        "content": "Please provide your final answer based on the information above."
                    })
            
            # First, get reasoning about how to approach the query (only for the first step)
            if step == 0 and self.tools:
                # Check if we already have reasoning from a previous run
                existing_reasoning = self.memory.get_last_reasoning()
                
                # Only get new reasoning if we don't have any or if the query is different
                if not existing_reasoning or not persistent_memory:
                    # Construct a reasoning prompt
                    reasoning_prompt = [
                        {"role": "system", "content": "You need to decide how to approach answering this query. Explain your thinking step by step."},
                        {"role": "user", "content": f"Query: {user_query}\n\nWhat tools would help answer this? How will you approach this problem?"}
                    ]
                    
                    try:
                        # Get the reasoning
                        reasoning = self.llm.generate(reasoning_prompt)
                        
                        if verbose:
                            print(f" Reasoning: {reasoning}")
                        
                        # Store reasoning in memory
                        self.memory.add_to_history({
                            "role": "assistant",
                            "content": f"Reasoning: {reasoning}",
                            "type": "reasoning",
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        if verbose:
                            print(f"Error getting reasoning: {str(e)}")
            
            # Generate a response with potential tool usage
            if self.tools:
                # Generate a response that might include tool calls
                if isinstance(self.llm, (OpenAILLM, CustomOpenAILLM)):
                    response = self.llm.generate_with_tools(conversation, self.tools)
                else:
                    # For other LLMs, convert to text format
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
                    response = self.llm.generate_with_tools(prompt, self.tools)
                
                if response["type"] == "tool_call":
                    # The LLM wants to use a tool
                    tool_name = response["tool"]
                    tool_args = response["args"]
                    tool_reasoning = response.get("reasoning", "No explicit reasoning provided")
                    
                    # Check if this is a duplicate tool call
                    is_duplicate = self.memory.is_duplicate_tool_call(tool_name, tool_args)
                    
                    # Check if we have a cached result
                    cache_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    cached_result = self.tool_cache.get(tool_name, tool_args)
                    
                    if verbose:
                        print(f"🛠️ Tool selected: {tool_name}")
                        print(f"🔍 Parameters: {json.dumps(tool_args, indent=2)}")
                        if is_duplicate:
                            print(f" This is a duplicate tool call")
                        if cached_result:
                            print(f" Using cached result")
                        print(f" Tool selection reasoning: {tool_reasoning}")
                    
                    # Add the reasoning to memory
                    self.memory.add_to_history({
                        "role": "assistant",
                        "content": f"Tool reasoning: {tool_reasoning}",
                        "type": "tool_reasoning",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Add the tool call to the conversation and memory
                    tool_call_message = f"I'll use the {tool_name} tool with these parameters: {json.dumps(tool_args)}"
                    conversation.append({
                        "role": "assistant",
                        "content": tool_call_message
                    })
                    
                    self.memory.add_to_history({
                        "role": "assistant",
                        "content": tool_call_message,
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": tool_args,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Use cached result if available, otherwise execute the tool
                    if cached_result:
                        if verbose:
                            print(f" Using cached result for {tool_name}")
                        tool_result = cached_result
                    else:
                        # Find the matching tool
                        called_tool = next((tool for tool in self.tools if tool.name == tool_name), None)
                        if called_tool:
                            tool_result = response["result"]  # Result from the response
                            # Cache the result for future use
                            self.tool_cache.set(tool_name, tool_args, tool_result)
                        else:
                            tool_result = {
                                "status": "error",
                                "error": f"Tool {tool_name} not found",
                                "timestamp": datetime.now().isoformat()
                            }
                    
                    # Handle success or error in tool result
                    if isinstance(tool_result, dict) and "status" in tool_result:
                        if tool_result["status"] == "success":
                            # Check if the result includes a visualization
                            has_visualization = "visualization" in tool_result
                            
                            if has_visualization:
                                visualization_data = tool_result["visualization"]
                                visualizations.append(visualization_data)
                                if verbose:
                                    print(f" Visualization generated")
                            
                            result_content = f"Tool {tool_name} returned: {json.dumps(tool_result['result'])}"
                            if verbose:
                                print(f" Result: {tool_result['result']}")
                        else:  # Error case
                            error_msg = tool_result.get("error", "Unknown error")
                            result_content = f"Tool {tool_name} failed with error: {error_msg}"
                            if verbose:
                                print(f" Error: {error_msg}")
                    else:
                        # Legacy format where the tool returns the result directly
                        result_content = f"Tool {tool_name} returned: {json.dumps(tool_result)}"
                        if verbose:
                            print(f" Result: {tool_result}")
                    
                    # Add the tool result to the conversation
                    conversation.append({
                        "role": "system",
                        "content": result_content
                    })
                    
                    # Record in memory
                    result_entry = {
                        "role": "system",
                        "content": result_content,
                        "type": "tool_result",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add visualization if present
                    if "visualization" in tool_result:
                        result_entry["visualization"] = tool_result["visualization"]
                    
                    self.memory.add_to_history(result_entry)
                    
                    # After getting the tool result, ask the model to reflect on what it learned
                    if step < max_steps - 1:  # Don't do this for the final step to save tokens
                        try:
                            reflection_prompt = [
                                {"role": "system", "content": "Based on the tool's result, reflect on what you've learned and how it helps answer the original query."},
                                {"role": "user", "content": f"Original query: {user_query}\nTool used: {tool_name}\nTool result: {tool_result}\n\nReflect on what you've learned:"}
                            ]
                            
                            reflection = self.llm.generate(reflection_prompt)
                            
                            if verbose:
                                print(f" Reflection: {reflection}")
                            
                            # Store reflection in memory
                            self.memory.add_to_history({
                                "role": "assistant",
                                "content": f"Reflection: {reflection}",
                                "type": "reflection",
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            if verbose:
                                print(f" Error getting reflection: {str(e)}")
                    
                    # Check if this completes a task (for multi-task queries)
                    if current_task and len(tasks) > 1:
                        # Check if this tool call completes the current task
                        if self.is_task_complete(current_task, tool_name):
                            task_idx, _ = self.task_tracker.get_next_incomplete_task()
                            self.task_tracker.mark_completed(task_idx)
                            if verbose:
                                print(f" Completed task {task_idx+1}: {current_task}")
                    
                    step += 1
                    continue
                
                elif response["type"] == "error":
                    # An error occurred during tool execution
                    error_message = response["content"]
                    
                    # Add the error to the conversation
                    conversation.append({
                        "role": "system",
                        "content": error_message
                    })
                    
                    # Record in memory
                    self.memory.add_to_history({
                        "role": "system",
                        "content": error_message,
                        "type": "error",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if verbose:
                        print(f" Error: {error_message}")
                    
                    step += 1
                    continue
                else:
                    # Direct text response
                    final_response = response["content"]
            else:
                # No tools, just get a text response
                if isinstance(self.llm, (OpenAILLM, CustomOpenAILLM)):
                    final_response = self.llm.generate_with_tools(conversation, [])["content"]
                else:
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
                    final_response = self.llm.generate(prompt)
            
            # Add the final response to conversation and memory
            conversation.append({
                "role": "assistant",
                "content": final_response
            })
            
            self.memory.add_to_history({
                "role": "assistant",
                "content": final_response,
                "type": "final_response",
                "timestamp": datetime.now().isoformat()
            })
            
            if verbose:
                print(f" Final response: {final_response}")
            
            # Check if we have completed all tasks
            if len(tasks) > 1:
                task_status = self.task_tracker.get_completion_status()
                
                # Mark all tasks as complete since we're providing a final answer
                for i in range(len(tasks)):
                    if i not in self.task_tracker.completed_tasks:
                        self.task_tracker.mark_completed(i)
                
                if verbose:
                    print(f" Tasks completed: {task_status['completed_tasks']}/{task_status['total_tasks']}")
            
            # We got a final answer, so we're done
            break
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Gather tool usage statistics
        tool_usage = {
            "total_calls": sum(tool.call_count for tool in self.tools),
            "calls_per_tool": {tool.name: tool.call_count for tool in self.tools},
            "execution_time_per_tool": {tool.name: tool.cumulative_execution_time for tool in self.tools},
            "cache_stats": self.tool_cache.get_stats()
        }
        
        # Prepare the return value based on requested format
        if return_trace:
            # Return structured response with trace
            return {
                "response": final_response,
                "history": self.memory.history,
                "trace": self.trace.format_trace(),
                "visualizations": visualizations,
                "token_usage": self.llm.get_token_usage() if hasattr(self.llm, "get_token_usage") else {},
                "tool_usage": tool_usage,
                "execution_time": execution_time,
                "all_tasks_completed": len(tasks) == 1 or self.task_tracker.all_tasks_completed()
            }
        else:
            # Return simple response and history
            return final_response, self.memory.history
    
    def save_memory(self, filename: str) -> None:
        """Save the agent's memory to a file."""
        self.memory.save_to_file(filename)
    
    def load_memory(self, filename: str) -> None:
        """Load the agent's memory from a file."""
        self.memory.load_from_file(filename)
        self.trace = Trace(self.memory)  # Reinitialize trace with the loaded memory

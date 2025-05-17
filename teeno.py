# tinychain.py - Enhanced minimalist LLM agent framework with transparency and caching
import json
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Generic, Tuple
import inspect
import re
import time
import hashlib
from datetime import datetime

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
    """Cache for storing and retrieving tool call results."""
    
    def __init__(self, max_size: int = 100, ttl: Optional[int] = None):
        """
        Initialize a new tool cache.
        
        Args:
            max_size: Maximum number of cached results to store
            ttl: Time-to-live in seconds for cache entries (None for no expiration)
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}  # {cache_key: (result, timestamp)}
        self.max_size = max_size
        self.ttl = ttl
    
    def _generate_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate a unique cache key for a tool call."""
        # Sort args to ensure consistent keys for same arguments
        sorted_args = json.dumps(args, sort_keys=True)
        key_string = f"{tool_name}:{sorted_args}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if it exists and is not expired."""
        key = self._generate_key(tool_name, args)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if the entry has expired
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                # Entry expired, remove it
                del self.cache[key]
                return None
            
            return result
        
        return None
    
    def set(self, tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache a tool call result."""
        key = self._generate_key(tool_name, args)
        
        # Implement LRU-like behavior if we're at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove the oldest entry (simple approach)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Store the result with the current timestamp
        self.cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
    
    def remove(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Remove a specific cached entry."""
        key = self._generate_key(tool_name, args)
        if key in self.cache:
            del self.cache[key]


class Tool:
    """Represents a callable tool for the LLM to use."""
    
    def __init__(self, name: str, func: Callable, description: str, cacheable: bool = True):
        self.name = name
        self.func = func
        self.description = description
        self.signature = self._get_signature()
        self.cacheable = cacheable  # Flag to indicate if this tool's results should be cached
        
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
        """Execute the tool with the provided arguments and handle caching."""
        # Extract cache from kwargs if present
        cache = kwargs.pop('cache', None)
        
        # Check if we should try to use cache
        if cache is not None and self.cacheable:
            # Try to get cached result
            cached_result = cache.get(self.name, kwargs)
            if cached_result is not None:
                # Add a flag to indicate this was a cache hit
                cached_result["cached"] = True
                return cached_result
        
        # No cache hit, execute the tool
        try:
            result = {
                "status": "success",
                "result": self.func(*args, **kwargs),
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
            
            # Store in cache if applicable
            if cache is not None and self.cacheable:
                cache.set(self.name, kwargs, result)
                
            return result
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
            
            # We don't cache errors
            return error_result


class Memory:
    """Enhanced memory store with better tracking of conversation and reasoning."""
    
    def __init__(self):
        self.storage = {}
        self.history = []
        self.tool_calls = []
        self.reasoning_steps = []
    
    def set(self, key: str, value: Any) -> None:
        """Store a value by key."""
        self.storage[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        return self.storage.get(key, default)
    
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the conversation history."""
        # Add a timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
            
        self.history.append(entry)
        
        # Track specialized entries
        if entry.get("type") == "reasoning":
            self.reasoning_steps.append(entry)
        elif entry.get("type") == "tool_call":
            self.tool_calls.append(entry)
    
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
    
    def get_reasoning_chain(self) -> List[str]:
        """Get the full chain of reasoning steps."""
        return [entry.get("content", "").replace("Reasoning:", "").strip() 
                for entry in self.history if entry.get("type") == "reasoning"]


class Chain:
    """Composes multiple operations into a single workflow."""
    
    def __init__(self):
        self.steps = []
        self.memory = Memory()
    
    def add(self, step_func: Callable[[Dict[str, Any], Memory], Dict[str, Any]]) -> 'Chain':
        """Add a step to the chain."""
        self.steps.append(step_func)
        return self
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all steps in the chain sequentially."""
        context = inputs
        
        for step in self.steps:
            context = step(context, self.memory)
            
        return context


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
            elif entry_type == "tool_call":
                lines.append(f"🛠️ TOOL CALL ({entry.get('tool', 'unknown')}): {content}")
            elif "Tool result:" in content:
                cached = " (CACHED)" if entry.get("cached", False) else ""
                lines.append(f"📊 RESULT{cached}: {content.replace('Tool result:', '').strip()}")
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
            elif entry_type == "tool_call":
                tool = entry.get("tool", "unknown")
                lines.append(f"## Step {current_step}: Tool Call - {tool}")
                lines.append(f"**Parameters:** {entry.get('args', {})}")
                current_step += 1
            elif "Tool result:" in content:
                cached = " (CACHED)" if entry.get("cached", False) else ""
                lines.append(f"### Result{cached}")
                lines.append(f"```\n{content.replace('Tool result:', '').strip()}\n```")
            elif role == "USER":
                lines.append(f"## Query")
                lines.append(f"> {content}")
            elif role == "ASSISTANT" and "Used tool:" not in content:
                lines.append(f"## Final Answer")
                lines.append(content)
                
            lines.append("")
            
        return "\n".join(lines)


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
    
    def generate(self, prompt: str) -> str:
        """Generate a text response using the OpenAI API."""
        # Handle both string and message list formats
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
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
            
            # Add bypass_cache parameter to tools that are cacheable
            for i, tool in enumerate(tools):
                if tool.cacheable:
                    tool_schemas[i]["function"]["parameters"]["properties"]["bypass_cache"] = {
                        "type": "boolean",
                        "description": "Set to true to bypass the cache and force a fresh call"
                    }
            
            # Make the API call with tool calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto"
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
                        "reasoning": reasoning
                    }
            
            # If no tool call or tool not found, return the text response
            return {
                "type": "text",
                "content": message.content
            }
            
        except Exception as e:
            # If function calling fails, fall back to the text-based approach
            # This makes our implementation model-agnostic
            return self._text_based_tool_calling(messages, tools)
    
    def _text_based_tool_calling(self, messages, tools):
        """Fall back to text-based tool calling when function calling is not supported."""
        # Add tool descriptions to the system message
        tools_desc = "\n".join([
            f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.signature)}\nCacheable: {tool.cacheable}"
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
  "param2": "value2",
  "bypass_cache": false  # Optional: Set to true to force a fresh tool call
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
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
                        "reasoning": reasoning
                    }
                except Exception as e:
                    # If tool execution fails, return an error
                    return {
                        "type": "error",
                        "content": f"I tried to use the {tool_name} tool but encountered an error: {str(e)}",
                        "error": str(e),
                        "tool": tool_name,
                        "args": arguments
                    }
        
        # If no tool call detected or parsing failed, return the text response
        return {
            "type": "text",
            "content": message_content
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


# For backward compatibility with the pasted code
CustomOpenAILLM = OpenAILLM


# =========== Agent Implementation ===========

class Agent:
    """Enhanced LLM-powered agent with transparency, reasoning, and caching capabilities."""
    
    def __init__(self, llm: LLM, system_prompt: Optional[str] = None,
                 cache_ttl: Optional[int] = 3600, cache_size: int = 100):
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.tools = []
        self.memory = Memory()
        self.trace = Trace(self.memory)
        self.cache = ToolCache(max_size=cache_size, ttl=cache_ttl)
        
    def add_tool(self, tool: Tool) -> 'Agent':
        """Add a tool for the agent to use."""
        self.tools.append(tool)
        return self
    
    def add_tools(self, tools: List[Tool]) -> 'Agent':
        """Add multiple tools for the agent to use."""
        self.tools.extend(tools)
        return self
    
    def clear_cache(self) -> None:
        """Clear all cached tool results."""
        self.cache.clear()

    def invalidate_tool_cache(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> None:
        """
        Invalidate cache entries for a specific tool.
        
        Args:
            tool_name: Name of the tool to invalidate cache for
            args: If provided, only invalidate cache for this specific set of arguments
        """
        if args is not None:
            # Remove specific cache entry
            self.cache.remove(tool_name, args)
        else:
            # Remove all entries for this tool
            # We'll need to iterate through all keys and check if they start with the tool name
            keys_to_remove = []
            for key in list(self.cache.cache.keys()):
                if key.startswith(self._generate_tool_key_prefix(tool_name)):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache.cache[key]
    
    def _generate_tool_key_prefix(self, tool_name: str) -> str:
        """Generate a prefix for cache keys for a specific tool."""
        # This is a simplified version, actual implementation would depend on
        # how _generate_key is implemented in ToolCache
        return hashlib.md5(tool_name.encode()).hexdigest()[:8]
    
    def run(self, user_query: str, max_steps: int = 5, verbose: bool = False, 
            return_trace: bool = False, use_cache: bool = True) -> Union[tuple, Dict[str, Any]]:
        """
        Run the agent to respond to a user query, using tools if necessary.
        
        Args:
            user_query: The user's question or request
            max_steps: Maximum number of reasoning/tool steps to take
            verbose: Whether to print detailed progress information
            return_trace: Whether to return the full execution trace
            use_cache: Whether to use cached tool results when available
            
        Returns:
            If return_trace is False: (final_response, history)
            If return_trace is True: {
                "response": final_response, 
                "history": history,
                "trace": formatted trace
            }
        """
        # Use the cache only if explicitly requested
        active_cache = self.cache if use_cache else None
        
        # Initialize conversation with system prompt
        conversation = [{
            "role": "system",
            "content": self.system_prompt
        }]
        
        # Add cache guidance to system prompt if using cache
        if use_cache:
            cache_guidance = """
You have access to cached results from previous tool calls.
When you see a tool result marked as "cached", it means this data was retrieved from cache rather than calling the tool again.
Consider whether cached data is still relevant for the current query before using it.
If you need fresh data, explicitly state that you want to ignore the cache for a specific tool call by setting bypass_cache to true.
"""
            conversation[0]["content"] += "\n" + cache_guidance
        
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
        
        while step < max_steps:
            # We will hit the max steps if the model keeps choosing to use tools
            if step == max_steps - 1:
                # For the last step, explicitly ask for a final answer
                conversation.append({
                    "role": "user",
                    "content": "Please provide your final answer based on the information above."
                })
            
            # First, get reasoning about how to approach the query
            if step == 0 and self.tools:  # Only for the first step
                # Construct a reasoning prompt
                reasoning_prompt = [
                    {"role": "system", "content": "You need to decide how to approach answering this query. Explain your thinking step by step."},
                    {"role": "user", "content": f"Query: {user_query}\n\nWhat tools would help answer this? How will you approach this problem?"}
                ]
                
                try:
                    # Get the reasoning
                    reasoning = self.llm.generate(reasoning_prompt)
                    
                    if verbose:
                        print(f"🧠 Reasoning: {reasoning}")
                    
                    # Store reasoning in memory
                    self.memory.add_to_history({
                        "role": "assistant",
                        "content": f"Reasoning: {reasoning}",
                        "type": "reasoning",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Error getting reasoning: {str(e)}")
            
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
                    
                    # Check if the LLM explicitly requested to bypass the cache
                    bypass_cache = False
                    if "bypass_cache" in tool_args:
                        bypass_cache = bool(tool_args.pop("bypass_cache"))
                    
                    # Find the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if tool:
                        # Use the cache if available and not bypassed
                        cache_to_use = None if bypass_cache else active_cache
                        
                        # Execute the tool (or get cached result)
                        tool_result = tool(cache=cache_to_use, **tool_args)
                        
                        # Check if we got a cached result
                        if verbose:
                            print(f"🛠️ Tool selected: {tool_name}")
                            print(f"🔍 Parameters: {json.dumps(tool_args, indent=2)}")
                            print(f"🧠 Tool selection reasoning: {tool_reasoning}")
                            if tool_result.get("cached", False):
                                print("♻️ Using cached result")
                        
                        # Add the reasoning to memory
                        self.memory.add_to_history({
                            "role": "assistant",
                            "content": f"Tool reasoning: {tool_reasoning}",
                            "type": "tool_reasoning",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Add the tool call to the conversation and memory
                        cache_status = " (requesting fresh data)" if bypass_cache else ""
                        tool_call_message = f"I'll use the {tool_name} tool{cache_status} with these parameters: {json.dumps(tool_args)}"
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
                        
                        # Add cache status to tool result message
                        cache_notice = " (from cache)" if tool_result.get("cached", False) else ""
                        
                        # Handle success or error in tool result
                        if isinstance(tool_result, dict) and "status" in tool_result:
                            if tool_result["status"] == "success":
                                result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result['result'])}"
                                if verbose:
                                    print(f"📊 Result: {tool_result['result']}")
                            else:  # Error case
                                error_msg = tool_result.get("error", "Unknown error")
                                result_content = f"Tool {tool_name} failed with error: {error_msg}"
                                if verbose:
                                    print(f"❌ Error: {error_msg}")
                        else:
                            # Legacy format where the tool returns the result directly
                            result_content = f"Tool {tool_name} returned{cache_notice}: {json.dumps(tool_result)}"
                            if verbose:
                                print(f"📊 Result: {tool_result}")
                        
                        # Add the tool result to the conversation
                        conversation.append({
                            "role": "system",
                            "content": result_content
                        })
                        
                        # Record in memory
                        self.memory.add_to_history({
                            "role": "system",
                            "content": result_content,
                            "type": "tool_result",
                            "cached": tool_result.get("cached", False),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # After getting the tool result, ask the model to reflect on what it learned
                        if step < max_steps - 1:  # Don't do this for the final step to save tokens
                            try:
                                reflection_prompt = [
                                    {"role": "system", "content": "Based on the tool's result, reflect on what you've learned and how it helps answer the original query."},
                                    {"role": "user", "content": f"Original query: {user_query}\nTool used: {tool_name}\nTool result: {tool_result}\n\nReflect on what you've learned:"}
                                ]
                                
                                reflection = self.llm.generate(reflection_prompt)
                                
                                if verbose:
                                    print(f"🧠 Reflection: {reflection}")
                                
                                # Store reflection in memory
                                self.memory.add_to_history({
                                    "role": "assistant",
                                    "content": f"Reflection: {reflection}",
                                    "type": "reflection",
                                    "timestamp": datetime.now().isoformat()
                                })
                            except Exception as e:
                                if verbose:
                                    print(f"⚠️ Error getting reflection: {str(e)}")
                        
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
                        print(f"❌ Error: {error_message}")
                    
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
                print(f"🤖 Final response: {final_response}")
            
            # We got a final answer, so we're done
            break
        
        # Prepare the return value based on requested format
        if return_trace:
            # Return structured response with trace
            return {
                "response": final_response,
                "history": self.memory.history,
                "trace": self.trace.format_trace(),
                "unique_tool_results": self.memory.get_unique_tool_results()
            }
        else:
            # Return simple response and history
            return final_response, self.memory.history

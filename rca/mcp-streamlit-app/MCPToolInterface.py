#!/usr/bin/env python3
"""
MCP Tool Interface - API Version
A simplified interface for testing MCP tools via function calls.
"""

import sys
import os
import json
import asyncio
import traceback
import ollama
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path to import mcp_sse_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP SSE client and model definitions
from mcp_sse_client.client import MCPClient, MCPConnectionError, MCPTimeoutError
from mcp_sse_client.llm_bridge.openai_bridge import OpenAIBridge
from mcp_sse_client.llm_bridge.anthropic_bridge import AnthropicBridge
from mcp_sse_client.llm_bridge.ollama_bridge import OllamaBridge
from mcp_sse_client.llm_bridge.models import (
    OPENAI_MODELS, DEFAULT_OPENAI_MODEL,
    ANTHROPIC_MODELS, DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_MODEL
)


class MCPToolInterface:
    """
    MCP Tool Interface for programmatic access to MCP tools.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "mcp_endpoint": "http://localhost:8080/sse",
        "llm_provider": "openai",  # "openai", "anthropic", "ollama"
        "api_keys": {
            "openai": os.environ.get("OPENAI_API_KEY", "lm-studio"),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
        },
        "ollama_config": {
            "model": "llama3.2:3b",
            "host": None  # None for default, or "http://localhost:11434"
        },
        "chat_mode": "auto",  # "auto", "chat", "tools"
        "connection_timeout": 30.0,
        "max_retries": 3
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP Tool Interface with optional configuration.

        Args:
            config: Optional configuration dictionary. Will merge with defaults.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self._merge_config(config)

        self.connected = False
        self.client = None
        self.llm_bridge = None
        self.tools = []
        self.connection_error = None

    def _merge_config(self, user_config: Dict[str, Any]):
        """Recursively merge user config with default config."""
        for key, value in user_config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    async def initialize(self) -> bool:
        """
        Initialize connection to MCP server and LLM provider.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        try:
            self.connection_error = None

            # Create MCP client
            self.client = MCPClient(
                self.config["mcp_endpoint"],
                timeout=self.config["connection_timeout"],
                max_retries=self.config["max_retries"]
            )

            # Test connection by listing tools
            self.tools = await self.client.list_tools()

            # Create LLM bridge based on provider
            self.llm_bridge = None
            provider = self.config["llm_provider"]

            if provider == "openai" and self.config["api_keys"]["openai"]:
                self.llm_bridge = OpenAIBridge(self.client, api_key=self.config["api_keys"]["openai"])
            elif provider == "anthropic" and self.config["api_keys"]["anthropic"]:
                self.llm_bridge = AnthropicBridge(self.client, api_key=self.config["api_keys"]["anthropic"])
            elif provider == "ollama":
                self.llm_bridge = OllamaBridge(
                    self.client,
                    model=self.config["ollama_config"]["model"],
                    host=self.config["ollama_config"]["host"]
                )

            self.connected = True
            return True

        except MCPConnectionError as e:
            self.connected = False
            self.connection_error = f"Connection failed: {e}"
            return False
        except MCPTimeoutError as e:
            self.connected = False
            self.connection_error = f"Connection timed out: {e}"
            return False
        except Exception as e:
            self.connected = False
            self.connection_error = f"Unexpected error: {e}"
            return False

    async def chat_with_llm_directly(self, messages: List[Dict[str, str]]) -> str:
        """Chat directly with LLM without tools."""
        # Extract the last user message for direct chat
        user_input = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_input = message.get("content", "")
                break

        if not user_input:
            return "No user message found in the conversation."

        provider = self.config["llm_provider"]

        if provider == "ollama":
            try:
                host = self.config["ollama_config"]["host"]
                client = ollama.AsyncClient(host=host)

                response = await client.chat(
                    model=self.config["ollama_config"]["model"],
                    messages=messages
                )

                return response.get('message', {}).get('content', 'No response received')

            except Exception as e:
                return f"Error chatting with Ollama: {e}"

        elif provider == "openai":
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=self.config["api_keys"]["openai"])

                response = await client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=messages
                )

                return response.choices[0].message.content

            except Exception as e:
                return f"Error chatting with OpenAI: {e}"

        elif provider == "anthropic":
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=self.config["api_keys"]["anthropic"])

                # Convert messages format for Anthropic
                anthropic_messages = []
                for msg in messages:
                    if msg["role"] in ["user", "assistant"]:
                        anthropic_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                response = await client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=1000,
                    messages=anthropic_messages
                )

                return response.content[0].text

            except Exception as e:
                return f"Error chatting with Anthropic: {e}"

        return "No LLM provider configured for direct chat."

    def extract_content_from_llm_response(self, llm_response):
        """Extract clean text content from different LLM provider response formats."""
        try:
            # OpenAI ChatCompletion object
            if hasattr(llm_response, 'choices') and llm_response.choices:
                content = llm_response.choices[0].message.content
                return content if content is not None else "No content received from OpenAI"

            # Anthropic Message object
            elif hasattr(llm_response, 'content') and llm_response.content:
                for content in llm_response.content:
                    if hasattr(content, 'type') and content.type == "text":
                        text = content.text
                        return text if text is not None else "No text content from Anthropic"
                first_content = str(llm_response.content[0])
                return first_content if first_content else "Empty content from Anthropic"

            # Ollama dict response
            elif isinstance(llm_response, dict):
                if 'message' in llm_response:
                    message = llm_response['message']
                    if isinstance(message, dict) and 'content' in message:
                        content = message['content']
                        return content if content is not None else "No content in Ollama message"
                content = llm_response.get('content', str(llm_response))
                return content if content else "Empty Ollama response"

            # String response
            elif isinstance(llm_response, str):
                return llm_response if llm_response else "Empty string response"

            # None response
            elif llm_response is None:
                return "No response received"

            # Fallback
            fallback = str(llm_response)
            return fallback if fallback else "Empty response object"

        except Exception as e:
            return f"Error extracting content: {e}"

    def format_tool_result(self, tool_result_content):
        """Format tool result content for better readability."""
        try:
            # Try to parse as JSON and format nicely
            if isinstance(tool_result_content, str) and tool_result_content.strip().startswith(('[', '{')):
                try:
                    parsed_json = json.loads(tool_result_content)

                    # Handle nested structure
                    if isinstance(parsed_json, dict) and 'text' in parsed_json:
                        inner_text = parsed_json['text']
                        if isinstance(inner_text, str) and inner_text.strip().startswith('['):
                            try:
                                inner_parsed = json.loads(inner_text)
                                parsed_json = inner_parsed
                            except json.JSONDecodeError:
                                return inner_text

                    # Special formatting for JIRA issues
                    if isinstance(parsed_json, list) and len(parsed_json) > 0:
                        if all(isinstance(item, dict) and 'key' in item for item in parsed_json):
                            formatted_issues = []
                            for issue in parsed_json:
                                issue_text = f"{issue.get('key', 'Unknown')}: {issue.get('summary', 'No summary')}"
                                if 'status' in issue:
                                    issue_text += f"\n  - Status: {issue['status']}"
                                if 'priority' in issue:
                                    issue_text += f"\n  - Priority: {issue['priority']}"
                                if 'assignee' in issue:
                                    issue_text += f"\n  - Assignee: {issue['assignee']}"
                                if 'created' in issue:
                                    issue_text += f"\n  - Created: {issue['created']}"
                                formatted_issues.append(issue_text)

                            return f"Found {len(parsed_json)} issues:\n\n" + "\n\n".join(formatted_issues)

                    # General JSON formatting
                    return json.dumps(parsed_json, indent=2, ensure_ascii=False)

                except json.JSONDecodeError:
                    pass

            return tool_result_content

        except Exception as e:
            return tool_result_content

    async def process_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Process a list of messages and return the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello, how are you?"}]

        Returns:
            str: The response from the MCP tool or LLM.
        """
        if not self.connected:
            return "❌ Not connected to MCP server. Please call initialize() first."

        if not messages:
            return "❌ No messages provided."

        # Extract the last user message for processing
        user_input = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_input = message.get("content", "")
                break

        if not user_input:
            return "❌ No user message found in the conversation."

        chat_mode = self.config["chat_mode"]

        # Chat mode: direct LLM conversation without tools
        if chat_mode == "chat":
            return await self.chat_with_llm_directly(messages)

        # Tools mode: always use tools
        elif chat_mode == "tools":
            if not self.llm_bridge:
                return "❌ No LLM bridge configured. Please check your configuration."

            try:
                result = await self.llm_bridge.process_query(user_input)

                if isinstance(result, dict):
                    llm_response = result.get("llm_response", {})
                    tool_call = result.get("tool_call")
                    tool_result = result.get("tool_result")

                    content = self.extract_content_from_llm_response(llm_response)
                    response_parts = []

                    if tool_call and tool_result:
                        if not content.startswith("No content received from"):
                            response_parts.append(content)

                        tool_name = tool_call.get('name', 'Unknown')
                        if tool_result.error_code == 0:
                            response_parts.append(f"🔧 Tool Used: {tool_name}")
                            formatted_result = self.format_tool_result(tool_result.content)
                            response_parts.append(f"Result:\n{formatted_result}")
                        else:
                            response_parts.append(f"❌ Tool Error: {tool_name} failed")
                            response_parts.append(f"Error: {tool_result.content}")
                    else:
                        response_parts.append(content)

                    return "\n".join(response_parts)
                else:
                    return self.extract_content_from_llm_response(result)

            except Exception as e:
                return f"Sorry, I encountered an error: {e}"

        # Auto mode: let LLM decide whether to use tools
        else:  # auto mode
            if not self.llm_bridge:
                return await self.chat_with_llm_directly(messages)

            try:
                result = await self.llm_bridge.process_query(user_input)

                if isinstance(result, dict):
                    llm_response = result.get("llm_response", {})
                    tool_call = result.get("tool_call")
                    tool_result = result.get("tool_result")

                    content = self.extract_content_from_llm_response(llm_response)
                    response_parts = []

                    if tool_call and tool_result:
                        if not content.startswith("No content received from"):
                            response_parts.append(content)

                        tool_name = tool_call.get('name', 'Unknown')
                        if tool_result.error_code == 0:
                            response_parts.append(f"🔧 Tool Used: {tool_name}")
                            formatted_result = self.format_tool_result(tool_result.content)
                            response_parts.append(f"Result:\n{formatted_result}")
                        else:
                            response_parts.append(f"❌ Tool Error: {tool_name} failed")
                            response_parts.append(f"Error: {tool_result.content}")
                    else:
                        response_parts.append(content)

                    return "\n".join(response_parts)
                else:
                    return self.extract_content_from_llm_response(result)

            except Exception as e:
                return f"Sorry, I encountered an error: {e}"

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MCP tool interface.

        Returns:
            dict: Status information including connection status, tools, etc.
        """
        return {
            "connected": self.connected,
            "connection_error": self.connection_error,
            "tools_count": len(self.tools),
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
            "config": {
                "mcp_endpoint": self.config["mcp_endpoint"],
                "llm_provider": self.config["llm_provider"],
                "chat_mode": self.config["chat_mode"]
            }
        }

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            try:
                await self.client.close()
            except:
                pass
        self.connected = False
        self.client = None
        self.llm_bridge = None


# Convenience function for single-use scenarios
async def process_mcp_messages(
        messages: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to process messages with MCP tools in a single call.

    Args:
        messages: List of message dictionaries
        config: Optional configuration dictionary

    Returns:
        str: The response from processing the messages
    """
    interface = MCPToolInterface(config)

    try:
        success = await interface.initialize()
        if not success:
            return f"❌ Failed to initialize MCP interface: {interface.connection_error}"

        response = await interface.process_messages(messages)
        return response

    finally:
        await interface.cleanup()
#!/usr/bin/env python3
"""
MCP Tool Wrapper - Simplified API for MCP Tool Integration
A clean interface for integrating MCP tools into other applications.
"""

import sys
import os
import json
import asyncio
import datetime
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


class MCPToolWrapper:
    """
    A wrapper class for MCP tools that provides a simple API for integration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MCP Tool Wrapper.

        Args:
            config: Configuration dictionary with the following optional keys:
                - mcp_endpoint: MCP server endpoint (default: "http://localhost:8080/sse")
                - llm_provider: "openai", "anthropic", or "ollama" (default: "openai")
                - api_keys: Dict with API keys for openai/anthropic
                - ollama_config: Dict with model and host for Ollama
                - chat_mode: "auto", "chat", or "tools" (default: "auto")
                - connection_timeout: Connection timeout in seconds (default: 30.0)
                - max_retries: Maximum connection retries (default: 3)
        """
        # Default configuration
        default_config = {
            "mcp_endpoint": "http://localhost:8080/sse",
            "llm_provider": "openai",
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", "lm-studio"),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
            },
            "ollama_config": {
                "model": "llama3.2:3b",
                "host": None
            },
            "chat_mode": "auto",
            "connection_timeout": 30.0,
            "max_retries": 3
        }

        # Merge user config with defaults
        self.config = default_config
        if config:
            self._deep_update(self.config, config)

        # Initialize instance variables
        self.connected = False
        self.client = None
        self.llm_bridge = None
        self.tools = []
        self.connection_error = None

    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    async def initialize(self) -> bool:
        """
        Initialize the MCP client and LLM bridge.

        Returns:
            bool: True if initialization successful, False otherwise
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
                self.llm_bridge = OpenAIBridge(
                    self.client,
                    api_key=self.config["api_keys"]["openai"]
                )
            elif provider == "anthropic" and self.config["api_keys"]["anthropic"]:
                self.llm_bridge = AnthropicBridge(
                    self.client,
                    api_key=self.config["api_keys"]["anthropic"]
                )
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

    async def process_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Process a list of messages and return the response.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "Hello, how are you?"}]

        Returns:
            str: Response from the LLM and/or tools
        """
        if not self.connected:
            return f"❌ Not connected to server. Error: {self.connection_error}"

        # Extract the latest user message (assuming that's what we want to process)


        return await self._process_user_input(messages)

    async def _process_user_input(self, user_input: List[Dict[str, str]]) -> str:
        """Process user input based on chat mode."""
        chat_mode = self.config["chat_mode"]

        # Chat mode: direct LLM conversation without tools
        if chat_mode == "chat":
            return await self._chat_with_llm_directly(user_input)

        # Tools mode: always use tools
        elif chat_mode == "tools":
            if not self.llm_bridge:
                return "❌ No LLM bridge configured. Please check your configuration."

            try:
                result = await self.llm_bridge.process_query(user_input)
                return self._format_llm_result(result)
            except Exception as e:
                return f"Sorry, I encountered an error: {e}"

        # Auto mode: let LLM decide whether to use tools
        else:  # auto mode
            if not self.llm_bridge:
                return await self._chat_with_llm_directly(user_input)

            try:
                result = await self.llm_bridge.process_query(user_input)
                return self._format_llm_result(result)
            except Exception as e:
                return f"Sorry, I encountered an error: {e}"

    async def _chat_with_llm_directly(self, user_input: List[Dict[str, str]]) -> str:
        """Chat directly with LLM without tools."""
        provider = self.config["llm_provider"]

        if provider == "ollama":
            try:
                host = self.config["ollama_config"]["host"]
                client = ollama.AsyncClient(host=host)

                response = await client.chat(
                    model=self.config["ollama_config"]["model"],
                    messages=user_input
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
                    messages=user_input
                )

                return response.choices[0].message.content

            except Exception as e:
                return f"Error chatting with OpenAI: {e}"

        elif provider == "anthropic":
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=self.config["api_keys"]["anthropic"])

                response = await client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=1000,
                    messages= user_input
                )

                return response.content[0].text

            except Exception as e:
                return f"Error chatting with Anthropic: {e}"

        return "No LLM provider configured for direct chat."

    def _format_llm_result(self, result: Any) -> str:
        """Format the result from LLM bridge."""
        if isinstance(result, dict):
            llm_response = result.get("llm_response", {})
            tool_call = result.get("tool_call")
            tool_result = result.get("tool_result")

            content = self._extract_content_from_llm_response(llm_response)
            response_parts = []

            if tool_call and tool_result:
                if not content.startswith("No content received from"):
                    response_parts.append(content)

                tool_name = tool_call.get('name', 'Unknown')
                if tool_result.error_code == 0:
                    response_parts.append(f"🔧 Tool Used: {tool_name}")
                    formatted_result = self._format_tool_result(tool_result.content)
                    response_parts.append(f"Result:\n{formatted_result}")
                else:
                    response_parts.append(f"❌ Tool Error: {tool_name} failed")
                    response_parts.append(f"Error: {tool_result.content}")
            else:
                response_parts.append(content)

            return "\n".join(response_parts)
        else:
            return self._extract_content_from_llm_response(result)

    def _extract_content_from_llm_response(self, llm_response: Any) -> str:
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

    def _format_tool_result(self, tool_result_content: Any) -> str:
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

            return str(tool_result_content)

        except Exception as e:
            return str(tool_result_content)

    def get_tools(self) -> List[Any]:
        """
        Get list of available tools.

        Returns:
            List of available tools
        """
        return self.tools

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.

        Returns:
            Dictionary with connection status details
        """
        return {
            "connected": self.connected,
            "error": self.connection_error,
            "tools_count": len(self.tools),
            "config": self.config
        }

    async def close(self):
        """Close connections and cleanup resources."""
        if self.client:
            # Add any cleanup code if needed
            pass
        self.connected = False


# Convenience function for quick usage
async def process_with_mcp(messages: List[Dict[str, str]], config: Dict[str, Any] = None) -> str:
    """
    Convenience function to process messages with MCP tools.

    Args:
        messages: List of message dictionaries
        config: Optional configuration dictionary

    Returns:
        str: Response from the system
    """
    wrapper = MCPToolWrapper(config)

    try:
        # Initialize the wrapper
        success = await wrapper.initialize()
        if not success:
            return f"❌ Failed to initialize: {wrapper.connection_error}"

        # Process the messages
        response = await wrapper.process_messages(messages)
        return response

    finally:
        # Cleanup
        await wrapper.close()
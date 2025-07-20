#!/usr/bin/env python3
"""
MCP Tool Tester - Command Line Version
A simplified command-line interface for testing MCP tools without Streamlit frontend.
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

# ================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ================================

# MCP Server Configuration
MCP_ENDPOINT = "http://localhost:8080/sse"

# LLM Provider Configuration - Choose one: "openai", "anthropic", "ollama"
LLM_PROVIDER = "openai"

# API Keys (only needed for OpenAI/Anthropic)
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", "lm-studio"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", "")
}

# Ollama Configuration (only used if LLM_PROVIDER is "ollama")
OLLAMA_CONFIG = {
    "model": "llama3.2:3b",  # Change this to your preferred Ollama model
    "host": None  # Set to "http://localhost:11434" or your custom Ollama host, None for default
}

# Chat Mode - Choose one: "auto", "chat", "tools"
# - "auto": LLM decides when to use tools
# - "chat": Direct conversation without tools
# - "tools": Always try to use tools
CHAT_MODE = "auto"

# Connection Settings
CONNECTION_TIMEOUT = 30.0
MAX_RETRIES = 3


# ================================
# END CONFIGURATION SECTION
# ================================

class MCPToolTester:
    def __init__(self):
        self.connected = False
        self.client = None
        self.llm_bridge = None
        self.tools = []
        self.connection_error = None

    async def fetch_ollama_models(self, host=None):
        """Asynchronously fetch available Ollama models from the server."""
        try:
            client = ollama.AsyncClient(host=host)
            models_info = await client.list()

            model_names = []

            # Direct extraction from ListResponse object
            if hasattr(models_info, 'models'):
                for model in models_info.models:
                    if hasattr(model, 'model') and model.model:
                        model_names.append(model.model)

            # Fallback extraction methods
            if not model_names:
                import re
                models_str = str(models_info)
                pattern = r"model='([^']+)'"
                model_names = re.findall(pattern, models_str)

                if not model_names and isinstance(models_info, dict) and 'models' in models_info:
                    model_names = [m.get('name', m.get('model', '')) for m in models_info.get('models', [])]
                elif not model_names and isinstance(models_info, list):
                    model_names = [m.get('name', m.get('model', '')) for m in models_info]

            # Filter out empty names
            model_names = [name for name in model_names if name]
            return model_names

        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    async def chat_with_llm_directly(self, user_input):
        """Chat directly with LLM without tools."""
        if LLM_PROVIDER == "ollama":
            try:
                host = OLLAMA_CONFIG["host"]
                client = ollama.AsyncClient(host=host)

                response = await client.chat(
                    model=OLLAMA_CONFIG["model"],
                    messages=[{"role": "user", "content": user_input}]
                )

                return response.get('message', {}).get('content', 'No response received')

            except Exception as e:
                return f"Error chatting with Ollama: {e}"

        elif LLM_PROVIDER == "openai":
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=API_KEYS["openai"])

                response = await client.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[{"role": "user", "content": user_input}]
                )

                return response.choices[0].message.content

            except Exception as e:
                return f"Error chatting with OpenAI: {e}"

        elif LLM_PROVIDER == "anthropic":
            try:
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=API_KEYS["anthropic"])

                response = await client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": user_input}]
                )

                return response.content[0].text

            except Exception as e:
                return f"Error chatting with Anthropic: {e}"

        return "No LLM provider configured for direct chat."

    async def connect_to_server(self):
        """Connect to MCP server and LLM provider."""
        try:
            self.connection_error = None

            # Create MCP client
            self.client = MCPClient(
                MCP_ENDPOINT,
                timeout=CONNECTION_TIMEOUT,
                max_retries=MAX_RETRIES
            )

            # Test connection by listing tools
            self.tools = await self.client.list_tools()

            # Create LLM bridge based on provider
            self.llm_bridge = None
            if LLM_PROVIDER == "openai" and API_KEYS["openai"]:
                self.llm_bridge = OpenAIBridge(self.client, api_key=API_KEYS["openai"])
            elif LLM_PROVIDER == "anthropic" and API_KEYS["anthropic"]:
                self.llm_bridge = AnthropicBridge(self.client, api_key=API_KEYS["anthropic"])
            elif LLM_PROVIDER == "ollama":
                self.llm_bridge = OllamaBridge(
                    self.client,
                    model=OLLAMA_CONFIG["model"],
                    host=OLLAMA_CONFIG["host"]
                )

            self.connected = True

            print(f"✅ Connected to {MCP_ENDPOINT}")
            if self.llm_bridge:
                print(f"✅ LLM bridge configured for {LLM_PROVIDER}")
                if LLM_PROVIDER == "ollama":
                    print(f"✅ Using Ollama model: {OLLAMA_CONFIG['model']}")
            if len(self.tools) > 0:
                print(f"✅ Found {len(self.tools)} available tools")
            else:
                print("⚠️ No tools found on the server")

            return True

        except MCPConnectionError as e:
            self.connected = False
            self.connection_error = f"Connection failed: {e}"
            print(f"❌ Connection failed: {e}")
            return False
        except MCPTimeoutError as e:
            self.connected = False
            self.connection_error = f"Connection timed out: {e}"
            print(f"❌ Connection timed out: {e}")
            return False
        except Exception as e:
            self.connected = False
            self.connection_error = f"Unexpected error: {e}"
            print(f"❌ Unexpected error: {e}")
            print(traceback.format_exc())
            return False

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

    async def process_user_message(self, user_input):
        """Process user message based on chat mode."""

        # Chat mode: direct LLM conversation without tools
        if CHAT_MODE == "chat":
            return await self.chat_with_llm_directly(user_input)

        # Tools mode: always use tools
        elif CHAT_MODE == "tools":
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
                return await self.chat_with_llm_directly(user_input)

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

    def print_configuration(self):
        """Print current configuration."""
        print("=" * 50)
        print("MCP Tool Tester - Configuration")
        print("=" * 50)
        print(f"MCP Endpoint: {MCP_ENDPOINT}")
        print(f"LLM Provider: {LLM_PROVIDER}")
        print(f"Chat Mode: {CHAT_MODE}")
        if LLM_PROVIDER == "ollama":
            print(f"Ollama Model: {OLLAMA_CONFIG['model']}")
            print(f"Ollama Host: {OLLAMA_CONFIG['host'] or 'default'}")
        elif LLM_PROVIDER in ["openai", "anthropic"]:
            key_status = "configured" if API_KEYS[LLM_PROVIDER] else "not set"
            print(f"{LLM_PROVIDER.upper()} API Key: {key_status}")
        print("=" * 50)

    def print_tools(self):
        """Print available tools."""
        if self.tools:
            print(f"\nAvailable Tools ({len(self.tools)}):")
            print("-" * 30)
            for i, tool in enumerate(self.tools, 1):
                print(f"{i}. {tool.name}")
                print(f"   Description: {tool.description}")
                if hasattr(tool, 'parameters') and tool.parameters:
                    print("   Parameters:")
                    for param in tool.parameters:
                        required = "Required" if param.required else "Optional"
                        print(f"     - {param.name} ({required}): {param.description}")
                print()
        else:
            print("\nNo tools available.")

    async def run_interactive(self):
        """Run interactive chat session."""
        print("\nMCP Tool Tester - Interactive Mode")
        print("Type 'quit' to exit, 'tools' to list tools, 'config' to show configuration")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n>>> ").strip()
                messages = [{"role": "user", "content": user_input}]

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'tools':
                    self.print_tools()
                    continue
                elif user_input.lower() == 'config':
                    self.print_configuration()
                    continue
                elif not user_input:
                    continue

                print("Processing...")
                response = await self.process_user_message(messages)
                print(f"\n{response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main function."""
    tester = MCPToolTester()

    # Print configuration
    tester.print_configuration()

    # Connect to server
    print("\nConnecting to server...")
    success = await tester.connect_to_server()

    if success:
        # Show available tools
        tester.print_tools()

        # Start interactive session
        await tester.run_interactive()
    else:
        print("Failed to connect to server. Please check your configuration.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
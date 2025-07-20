#!/usr/bin/env python3
"""
Example usage of the MCP Tool Wrapper
Demonstrates how to use the wrapper in your own applications.
"""

import asyncio
from MCPToolWrapper import MCPToolWrapper, process_with_mcp


async def example_basic_usage():
    """Basic usage example with default configuration."""
    print("=== Basic Usage Example ===")

    messages = [{"role": "user", "content": "查询纽约天气"}]

    # Using the convenience function
    response = await process_with_mcp(messages)
    print("Response:", response)
    print()
#
#
# async def example_custom_config():
#     """Example with custom configuration."""
#     print("=== Custom Configuration Example ===")
#
#     # Custom configuration
#     config = {
#         "mcp_endpoint": "http://localhost:8080/sse",
#         "llm_provider": "ollama",  # or "openai", "anthropic"
#         "ollama_config": {
#             "model": "llama3.2:3b",
#             "host": None  # Use default
#         },
#         "chat_mode": "auto"  # or "chat", "tools"
#     }
#
#     messages = [
#         {"role": "user", "content": "What's the weather like today?"}
#     ]
#
#     # Using the convenience function with custom config
#     response = await process_with_mcp(messages, config)
#     print("Response:", response)
#     print()
#
#
# async def example_class_usage():
#     """Example using the wrapper class directly for more control."""
#     print("=== Class Usage Example ===")
#
#     # Create wrapper with custom config
#     config = {
#         "llm_provider": "openai",
#         "api_keys": {
#             "openai": "your-api-key-here"  # or use environment variable
#         },
#         "chat_mode": "auto"
#     }
#
#     wrapper = MCPToolWrapper(config)
#
#     try:
#         # Initialize
#         success = await wrapper.initialize()
#         if not success:
#             print(f"Failed to initialize: {wrapper.connection_error}")
#             return
#
#         # Check connection status
#         status = wrapper.get_connection_status()
#         print(f"Connected: {status['connected']}")
#         print(f"Available tools: {status['tools_count']}")
#
#         # Get available tools
#         tools = wrapper.get_tools()
#         if tools:
#             print("Available tools:")
#             for tool in tools:
#                 print(f"  - {tool.name}: {tool.description}")
#
#         # Process multiple conversations
#         conversations = [
#             [{"role": "user", "content": "Can you search for recent news about AI?"}],
#             [{"role": "user", "content": "What tools do you have available?"}],
#             [{"role": "user", "content": "Tell me a joke"}]
#         ]
#
#         for i, messages in enumerate(conversations, 1):
#             print(f"\n--- Conversation {i} ---")
#             response = await wrapper.process_messages(messages)
#             print("Response:", response)
#
#     finally:
#         # Always close the wrapper
#         await wrapper.close()
#
#
# async def example_error_handling():
#     """Example demonstrating error handling."""
#     print("=== Error Handling Example ===")
#
#     # Configuration with invalid endpoint
#     config = {
#         "mcp_endpoint": "http://invalid-endpoint:9999/sse",
#         "connection_timeout": 5.0  # Short timeout for demo
#     }
#
#     wrapper = MCPToolWrapper(config)
#
#     try:
#         success = await wrapper.initialize()
#         if not success:
#             status = wrapper.get_connection_status()
#             print(f"Connection failed as expected: {status['error']}")
#
#         # Try to process messages anyway (will return error message)
#         messages = [{"role": "user", "content": "Hello"}]
#         response = await wrapper.process_messages(messages)
#         print("Response:", response)
#
#     finally:
#         await wrapper.close()
#
#
# async def example_different_modes():
#     """Example showing different chat modes."""
#     print("=== Different Chat Modes Example ===")
#
#     modes = ["auto", "chat", "tools"]
#     messages = [{"role": "user", "content": "What can you help me with?"}]
#
#     for mode in modes:
#         print(f"\n--- Mode: {mode} ---")
#         config = {
#             "chat_mode": mode,
#             "llm_provider": "ollama"  # Using Ollama for this example
#         }
#
#         try:
#             response = await process_with_mcp(messages, config)
#             print("Response:", response)
#         except Exception as e:
#             print(f"Error in {mode} mode: {e}")


async def main():
    """Run all examples."""
    print("MCP Tool Wrapper Examples")
    print("=" * 50)

    examples = [
        example_basic_usage,
    ]

    for example in examples:
        try:
            await example()
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
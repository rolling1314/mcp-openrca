#!/usr/bin/env python3
"""
Direct test runner for MCP Tool Interface (without pytest)
"""

import asyncio
import json
from mcp_tool_interface import MCPToolInterface, process_mcp_messages


async def test_basic_usage():
    """Test basic usage of the MCP interface."""
    print("=" * 50)
    print("Testing Basic MCP Interface Usage")
    print("=" * 50)

    # Test messages
    messages = [
        {"role": "user", "content": "Hello! Can you help me with some tasks?"}
    ]

    # Use the convenience function
    response = await process_mcp_messages(messages)
    print(f"Response: {response}")
    print()


async def test_custom_config():
    """Test with custom configuration."""
    print("=" * 50)
    print("Testing Custom Configuration")
    print("=" * 50)

    # Custom config - using Ollama with different model
    custom_config = {
        "llm_provider": "ollama",
        "ollama_config": {
            "model": "llama3.2:1b",  # Smaller model for faster testing
            "host": None
        },
        "chat_mode": "auto"
    }

    messages = [
        {"role": "user", "content": "What tools do you have available?"}
    ]

    response = await process_mcp_messages(messages, custom_config)
    print(f"Response: {response}")
    print()


async def test_conversation_flow():
    """Test a conversation flow with multiple messages."""
    print("=" * 50)
    print("Testing Conversation Flow")
    print("=" * 50)

    interface = MCPToolInterface()

    try:
        # Initialize
        success = await interface.initialize()
        if not success:
            print(f"Failed to initialize: {interface.connection_error}")
            return

        # Show status
        status = interface.get_status()
        print(f"Connected: {status['connected']}")
        print(f"Tools available: {status['tools_count']}")
        if status['tools']:
            print("Available tools:")
            for tool in status['tools']:
                print(f"  - {tool['name']}: {tool['description']}")
        print()

        # Test conversation
        conversations = [
            [{"role": "user", "content": "Hi there!"}],
            [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
                {"role": "user", "content": "Can you search for some information about Python programming?"}
            ],
            [{"role": "user", "content": "What's the current time?"}]
        ]

        for i, messages in enumerate(conversations, 1):
            print(f"--- Conversation {i} ---")
            print(f"Messages: {json.dumps(messages, indent=2)}")
            response = await interface.process_messages(messages)
            print(f"Response: {response}")
            print()

    finally:
        await interface.cleanup()


async def test_different_modes():
    """Test different chat modes."""
    print("=" * 50)
    print("Testing Different Chat Modes")
    print("=" * 50)

    modes = ["chat", "auto", "tools"]
    test_message = [{"role": "user", "content": "Can you help me search for recent news about AI?"}]

    for mode in modes:
        print(f"--- Testing {mode.upper()} mode ---")

        config = {"chat_mode": mode}

        try:
            response = await process_mcp_messages(test_message, config)
            print(f"Response: {response}")

        except Exception as e:
            print(f"Error in {mode} mode: {e}")

        print()


async def test_error_handling():
    """Test error handling scenarios."""
    print("=" * 50)
    print("Testing Error Handling")
    print("=" * 50)

    # Test with invalid endpoint
    invalid_config = {
        "mcp_endpoint": "http://invalid-endpoint:9999/sse"
    }

    messages = [{"role": "user", "content": "This should fail"}]

    response = await process_mcp_messages(messages, invalid_config)
    print(f"Invalid endpoint response: {response}")
    print()

    # Test with empty messages
    response = await process_mcp_messages([])
    print(f"Empty messages response: {response}")
    print()

    # Test with no user message
    response = await process_mcp_messages([{"role": "system", "content": "System message only"}])
    print(f"No user message response: {response}")
    print()


async def interactive_test():
    """Interactive testing mode."""
    print("=" * 50)
    print("Interactive Testing Mode")
    print("=" * 50)
    print("Type 'quit' to exit, or enter messages to test")

    interface = MCPToolInterface()

    try:
        success = await interface.initialize()
        if not success:
            print(f"Failed to initialize: {interface.connection_error}")
            return

        status = interface.get_status()
        print(f"Connected successfully! {status['tools_count']} tools available.")

        conversation_history = []

        while True:
            user_input = input("\n>>> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'status':
                status = interface.get_status()
                print(json.dumps(status, indent=2))
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared.")
                continue
            elif not user_input:
                continue

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})

            print("Processing...")
            response = await interface.process_messages(conversation_history)
            print(f"\n{response}")

            # Add assistant response to conversation
            conversation_history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        await interface.cleanup()


async def main():
    """Main test function."""
    print("MCP Tool Interface Test Suite")
    print("=" * 50)

    tests = [
        ("Basic Usage", test_basic_usage),
        ("Custom Configuration", test_custom_config),
        ("Conversation Flow", test_conversation_flow),
        ("Different Modes", test_different_modes),
        ("Error Handling", test_error_handling)
    ]

    # Run basic tests
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 50)

    # Ask if user wants to run interactive test
    try:
        run_interactive = input("\nRun interactive test? (y/n): ").lower().startswith('y')
        if run_interactive:
            await interactive_test()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
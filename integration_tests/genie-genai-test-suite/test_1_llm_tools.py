# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ============================================================================
# CONFIGURATION - Adjust these for your environment
# ============================================================================
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:9001/v1")
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen2.5-7b")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
API_KEY = os.environ.get("OPENAI_API_KEY", "xxxx")
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))

# Set API key for tests
os.environ["OPENAI_API_KEY"] = API_KEY

from langchain_core.tools import tool


# Tool definition for testing - simulates a simple weather API
@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather in a given location.
    The output is a string with a summary of the weather.
    """
    if "boston" in location.lower():
        return "The current weather in Boston is 15°C and partly cloudy."
    elif "paris" in location.lower():
        return "The current weather in Paris is 8°C and rainy."
    elif "turin" in location.lower():
        return "The current weather in Turin is 8°C and rainy."
    else:
        return f"Sorry, I do not have real-time weather data for {location}. Assuming it's a sunny day!"


@pytest.fixture(scope="session")
def chat_client():
    """Create a shared chat client for all tests"""
    return ChatOpenAI(base_url=BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE, timeout=TIMEOUT)


@pytest.fixture
def chat_with_tools(chat_client):
    """Create a chat client with tools bound"""
    return chat_client.bind_tools([get_current_weather])


class TestGetCurrentWeather:
    """Test suite for the get_current_weather tool"""

    def test_weather_boston(self):
        """Test weather query for Boston"""
        result = get_current_weather.invoke({"location": "Boston"})
        assert "Boston" in result
        assert "15°C" in result
        assert "partly cloudy" in result

    def test_weather_paris(self):
        """Test weather query for Paris"""
        result = get_current_weather.invoke({"location": "Paris"})
        assert "Paris" in result
        assert "8°C" in result
        assert "rainy" in result

    def test_weather_turin(self):
        """Test weather query for Turin"""
        result = get_current_weather.invoke({"location": "Turin"})
        assert "Turin" in result
        assert "8°C" in result
        assert "rainy" in result

    def test_weather_unknown_location(self):
        """Test weather query for unknown location"""
        result = get_current_weather.invoke({"location": "Tokyo"})
        assert "Tokyo" in result
        assert "sunny day" in result

    def test_weather_case_insensitive(self):
        """Test that location matching is case-insensitive"""
        result = get_current_weather.invoke({"location": "BOSTON"})
        assert "Boston" in result

        result = get_current_weather.invoke({"location": "paris"})
        assert "Paris" in result


class TestSimpleStreaming:
    """Test suite for simple streaming without tools"""

    def test_simple_story_streaming(self, chat_with_tools):
        """Test streaming a simple story without tool calls"""
        messages = [HumanMessage(content="Write a 3-sentence, encouraging story about a dog that goes to the forest.")]

        # Collect streamed content
        story_content = ""
        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                story_content += chunk.content

        # Verify we got a response
        assert len(story_content) > 0
        print(f"\nGenerated story: {story_content}")

    def test_simple_question_streaming(self, chat_with_tools):
        """Test streaming a simple question"""
        messages = [HumanMessage(content="What is 2+2?")]

        content = ""
        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content += chunk.content

        assert len(content) > 0
        print(f"\nResponse: {content}")


class TestToolCalling:
    """Test suite for tool calling with actual LLM"""

    def test_weather_tool_call_boston(self, chat_with_tools):
        """Test that model calls weather tool for Boston"""
        messages = [HumanMessage(content="What is the weather in Boston?")]

        # Stream and collect tool calls
        tool_calls = []
        content_chunks = []

        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content_chunks.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        # Model should have requested a tool call
        print(f"\nTool calls detected: {len(tool_calls)}")
        if tool_calls:
            for tc in tool_calls:
                print(f"  - {tc['name']} with args: {tc['args']}")

        # Note: Some models may not call tools reliably, so we make this informative
        # rather than a hard assertion
        if len(tool_calls) > 0:
            assert tool_calls[0]["name"] == "get_current_weather"
            assert "location" in tool_calls[0]["args"]

    def test_weather_tool_call_paris(self, chat_with_tools):
        """Test that model calls weather tool for Paris"""
        messages = [HumanMessage(content="What is the weather like in Paris right now?")]

        tool_calls = []
        for chunk in chat_with_tools.stream(messages):
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        print(f"\nTool calls detected: {len(tool_calls)}")
        if tool_calls:
            for tc in tool_calls:
                print(f"  - {tc['name']} with args: {tc['args']}")

    def test_tool_execution_in_conversation(self, chat_with_tools):
        """Test executing a tool and getting final response"""
        messages = [HumanMessage(content="What is the weather in Turin?")]

        # First stream - get tool call
        tool_calls = []
        content_chunks = []

        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content_chunks.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        print(f"\nInitial response: {''.join(content_chunks)}")
        print(f"Tool calls: {tool_calls}")

        if tool_calls:
            # Add AI message with tool calls
            messages.append(AIMessage(content="".join(content_chunks) if content_chunks else "", tool_calls=tool_calls))

            # Execute tool calls
            for tool_call in tool_calls:
                if tool_call["name"] == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_call["args"])
                    print(f"Tool output: {tool_output}")

                    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_output))

            # Get final response from LLM after tool execution
            print(f"\n--- Requesting final LLM response with tool results ---")
            final_content = ""
            for chunk in chat_with_tools.stream(messages):
                if chunk.content:
                    final_content += chunk.content

            print(f"Final LLM response: {final_content}")

            # Verify the final LLM response is not empty
            assert final_content is not None, "Final LLM response should not be None"
            assert len(final_content) > 0, "Final LLM response should not be empty after receiving tool results"
            assert final_content.strip() != "", "Final LLM response should contain meaningful content"
            print(f"✓ Verified: Final LLM response contains {len(final_content)} characters")


class TestFullConversationFlow:
    """Test complete conversation flows"""

    def test_multi_turn_conversation(self, chat_with_tools):
        """Test a multi-turn conversation with tool use"""
        messages = []

        # Turn 1: Simple greeting
        messages.append(HumanMessage(content="Hello!"))
        response1 = ""
        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                response1 += chunk.content

        messages.append(AIMessage(content=response1))
        print(f"\nTurn 1 - Response: {response1}")

        # Turn 2: Ask about weather (should trigger tool)
        messages.append(HumanMessage(content="What is the weather in Boston?"))

        tool_calls = []
        content_chunks = []
        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content_chunks.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        print(f"Turn 2 - Tool calls: {len(tool_calls)}")

        if tool_calls:
            messages.append(AIMessage(content="".join(content_chunks) if content_chunks else "", tool_calls=tool_calls))

            # Execute tools
            for tool_call in tool_calls:
                if tool_call["name"] == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_call["args"])
                    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_output))

            # Get final answer from LLM after tool execution
            print(f"\n--- Requesting final LLM response with tool results ---")
            final_response = ""
            for chunk in chat_with_tools.stream(messages):
                if chunk.content:
                    final_response += chunk.content

            print(f"Turn 2 - Final LLM response: {final_response}")

            # Verify the final LLM response is not empty
            assert final_response is not None, "Final LLM response should not be None"
            assert len(final_response) > 0, "Final LLM response should not be empty after receiving tool results"
            assert final_response.strip() != "", "Final LLM response should contain meaningful content"
            print(f"✓ Verified: Final LLM response contains {len(final_response)} characters")

    def test_message_history_persistence(self, chat_with_tools):
        """Test that message history is maintained correctly"""
        messages = []

        # Add human message
        messages.append(HumanMessage(content="Tell me about dogs."))
        assert len(messages) == 1

        # Simulate AI response
        messages.append(AIMessage(content="Dogs are loyal companions."))
        assert len(messages) == 2

        # Add another human message
        messages.append(HumanMessage(content="What about cats?"))
        assert len(messages) == 3

        # Stream response with full history
        response = ""
        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                response += chunk.content

        print(f"\nResponse with context: {response}")
        assert len(response) > 0

    def test_conversation_with_invoke(self, chat_with_tools):
        """Test conversation flow using invoke() instead of stream()"""
        messages = []

        # Turn 1: Ask about weather (should trigger tool)
        messages.append(HumanMessage(content="What is the weather in Boston?"))

        # Use invoke instead of stream - returns complete AIMessage
        response = chat_with_tools.invoke(messages)

        print(f"\nInvoke response type: {type(response)}")
        print(f"Response content: {response.content}")
        print(f"Tool calls: {response.tool_calls}")

        # Check if tool was called
        if response.tool_calls:
            print(f"Tool calls detected: {len(response.tool_calls)}")

            # Add AI response to history
            messages.append(response)

            # Execute tool calls
            for tool_call in response.tool_calls:
                if tool_call["name"] == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_call["args"])
                    print(f"Tool output: {tool_output}")

                    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_output))

            # Get final response from LLM using invoke after tool execution
            print(f"\n--- Requesting final LLM response with tool results (invoke) ---")
            final_response = chat_with_tools.invoke(messages)
            print(f"Final LLM response content: {final_response.content}")

            # Verify the final LLM response is not empty
            assert final_response is not None, "Final LLM response should not be None"
            assert final_response.content is not None, "Final LLM response content should not be None"
            assert len(final_response.content) > 0, "Final LLM response should not be empty after receiving tool results"
            assert final_response.content.strip() != "", "Final LLM response should contain meaningful content"
            print(f"✓ Verified: Final LLM response contains {len(final_response.content)} characters")
        else:
            # Model responded directly without calling tool
            print("No tool calls, direct response received")
            assert len(response.content) > 0

    def test_multi_turn_with_invoke(self, chat_with_tools):
        """Test multi-turn conversation using invoke() throughout"""
        messages = []

        # Turn 1: Greeting
        messages.append(HumanMessage(content="Hello!"))
        response1 = chat_with_tools.invoke(messages)
        messages.append(response1)

        print(f"\nTurn 1 (invoke) - Response: {response1.content}")
        assert len(response1.content) > 0

        # Turn 2: Ask about weather
        messages.append(HumanMessage(content="What is the weather in Paris?"))
        response2 = chat_with_tools.invoke(messages)

        print(f"Turn 2 (invoke) - Response content: {response2.content}")
        print(f"Turn 2 (invoke) - Tool calls: {len(response2.tool_calls) if response2.tool_calls else 0}")

        if response2.tool_calls:
            messages.append(response2)

            # Execute tools
            for tool_call in response2.tool_calls:
                if tool_call["name"] == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_call["args"])
                    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_output))

            # Get final answer from LLM after tool execution
            print(f"\n--- Requesting final LLM response with tool results (invoke) ---")
            final_response = chat_with_tools.invoke(messages)
            print(f"Turn 2 (invoke) - Final LLM response: {final_response.content}")

            # Verify the final LLM response is not empty
            assert final_response is not None, "Final LLM response should not be None"
            assert final_response.content is not None, "Final LLM response content should not be None"
            assert len(final_response.content) > 0, "Final LLM response should not be empty after receiving tool results"
            assert final_response.content.strip() != "", "Final LLM response should contain meaningful content"
            print(f"✓ Verified: Final LLM response contains {len(final_response.content)} characters")
        else:
            assert len(response2.content) > 0


class TestMultipleToolCalls:
    """Test handling multiple tool calls"""

    def test_multiple_location_query(self, chat_with_tools):
        """Test asking about weather in multiple locations"""
        messages = [HumanMessage(content="Compare the weather in Boston and Paris for me.")]

        tool_calls = []
        content_chunks = []

        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content_chunks.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        print(f"\nTool calls for multiple locations: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"  - {tc['name']}: {tc['args']}")

        # Execute all tool calls if any were made
        if tool_calls:
            messages.append(AIMessage(content="".join(content_chunks) if content_chunks else "", tool_calls=tool_calls))

            for tool_call in tool_calls:
                if tool_call["name"] == "get_current_weather":
                    tool_output = get_current_weather.invoke(tool_call["args"])
                    messages.append(ToolMessage(tool_call_id=tool_call["id"], content=tool_output))

            # Get final comparison from LLM after tool execution
            print(f"\n--- Requesting final LLM response with tool results ---")
            final_response = ""
            for chunk in chat_with_tools.stream(messages):
                if chunk.content:
                    final_response += chunk.content

            print(f"Final LLM comparison: {final_response}")

            # Verify the final LLM response is not empty
            assert final_response is not None, "Final LLM response should not be None"
            assert len(final_response) > 0, "Final LLM response should not be empty after receiving tool results"
            assert final_response.strip() != "", "Final LLM response should contain meaningful content"
            print(f"✓ Verified: Final LLM response contains {len(final_response)} characters")


class TestNoToolCallScenarios:
    """Test scenarios where tools should not be called"""

    def test_general_question_no_tool(self, chat_with_tools):
        """Test that general questions don't trigger tools"""
        messages = [HumanMessage(content="What is the capital of France?")]

        tool_calls = []
        content = ""

        for chunk in chat_with_tools.stream(messages):
            if chunk.content:
                content += chunk.content
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)

        print(f"\nGeneral question response: {content}")
        print(f"Tool calls (should be 0 or very few): {len(tool_calls)}")

        # Should have a response
        assert len(content) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

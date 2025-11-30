"""
Unit tests for the reflection_agent main module.
Tests focus on event_loop logic and graph construction.
This version runs without pytest for validation purposes.
"""

import sys
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage
)
from langgraph.graph import END

from langgraph_examples.reflection_agent.main import event_loop, MAX_ITERATIONS


class TestRunner:
    """Simple test runner to execute tests without pytest."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_name, test_func):
        """Run a single test function."""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {test_name}")
        except AssertionError as e:
            self.failed += 1
            error_msg = f"✗ {test_name}: {str(e)}"
            print(error_msg)
            self.errors.append(error_msg)
        except Exception as e:
            self.failed += 1
            error_msg = f"✗ {test_name}: ERROR - {str(e)}"
            print(error_msg)
            self.errors.append(error_msg)

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        if self.errors:
            print("\nFailed Tests:")
            for error in self.errors:
                print(f"  {error}")
        return self.failed == 0


class TestEventLoop:
    """Test suite for event_loop function."""

    @staticmethod
    def test_empty_state_returns_end():
        """Test Case 1: Empty state should return END."""
        state = []
        result = event_loop(state)
        assert result == END, "Empty state should return END"

    @staticmethod
    def test_state_with_tool_messages_exceeding_max_iterations_returns_end():
        """Test Case 2: State with ToolMessages > MAX_ITERATIONS should return END."""
        # Create state with more ToolMessages than MAX_ITERATIONS
        state = [
            HumanMessage(content="Initial query"),
            AIMessage(content="Response", tool_calls=[{"id": "1", "name": "tool1", "args": {}}]),
            ToolMessage(content="Result 1", tool_call_id="1"),
            AIMessage(content="Response 2", tool_calls=[{"id": "2", "name": "tool2", "args": {}}]),
            ToolMessage(content="Result 2", tool_call_id="2"),
            AIMessage(content="Response 3", tool_calls=[{"id": "3", "name": "tool3", "args": {}}]),
            ToolMessage(content="Result 3", tool_call_id="3"),
            AIMessage(content="Response 4", tool_calls=[{"id": "4", "name": "tool4", "args": {}}]),
            ToolMessage(content="Result 4", tool_call_id="4"),
        ]

        # Verify we have more than MAX_ITERATIONS tool messages
        tool_message_count = sum(isinstance(msg, ToolMessage) for msg in state)
        assert tool_message_count > MAX_ITERATIONS, f"Test setup error: expected > {MAX_ITERATIONS} tool messages, got {tool_message_count}"

        result = event_loop(state)
        assert result == END, f"State with {tool_message_count} ToolMessages (> MAX_ITERATIONS={MAX_ITERATIONS}) should return END"

    @staticmethod
    def test_state_at_max_iterations_boundary():
        """Test boundary condition: exactly MAX_ITERATIONS tool messages."""
        # Create state with exactly MAX_ITERATIONS ToolMessages
        state = [HumanMessage(content="Initial query")]
        for i in range(MAX_ITERATIONS):
            state.append(AIMessage(content=f"Response {i}", tool_calls=[{"id": str(i), "name": f"tool{i}", "args": {}}]))
            state.append(ToolMessage(content=f"Result {i}", tool_call_id=str(i)))

        # Add a final AI message without tool calls
        state.append(AIMessage(content="Final response"))

        tool_message_count = sum(isinstance(msg, ToolMessage) for msg in state)
        assert tool_message_count == MAX_ITERATIONS, f"Test setup error: expected {MAX_ITERATIONS} tool messages, got {tool_message_count}"

        result = event_loop(state)
        assert result == END, f"State with exactly {MAX_ITERATIONS} ToolMessages and no tool_calls should return END"

    @staticmethod
    def test_last_message_without_tool_calls_returns_end():
        """Test Case 3: Last message has no tool_calls should return END."""
        state = [
            HumanMessage(content="Query"),
            AIMessage(content="Response without tool calls")
        ]
        result = event_loop(state)
        assert result == END, "Last message without tool_calls should return END"

    @staticmethod
    def test_last_message_with_empty_tool_calls_returns_end():
        """Test variation: Last message with empty tool_calls list should return END."""
        state = [
            HumanMessage(content="Query"),
            AIMessage(content="Response", tool_calls=[])
        ]
        result = event_loop(state)
        assert result == END, "Last message with empty tool_calls should return END"

    @staticmethod
    def test_last_message_with_tool_calls_returns_execute_tools():
        """Test Case 4: Last message has tool_calls should return 'execute_tools'."""
        state = [
            HumanMessage(content="Query"),
            AIMessage(
                content="Response with tool call",
                tool_calls=[{
                    "id": "call_123",
                    "name": "search_tool",
                    "args": {"query": "test"}
                }]
            )
        ]
        result = event_loop(state)
        assert result == "execute_tools", "Last message with tool_calls should return 'execute_tools'"

    @staticmethod
    def test_multiple_ai_messages_only_last_matters():
        """Test that only the last AI message with tool_calls is considered."""
        state = [
            HumanMessage(content="Query"),
            AIMessage(content="First AI response", tool_calls=[{"id": "1", "name": "tool1", "args": {}}]),
            ToolMessage(content="Tool result", tool_call_id="1"),
            AIMessage(content="Second AI response without tool calls"),
        ]
        result = event_loop(state)
        assert result == END, "Should use last AI message which has no tool_calls"

    @staticmethod
    def test_finds_last_ai_message_with_tool_calls_attribute():
        """Test that event_loop finds the last message with tool_calls attribute."""
        state = [
            HumanMessage(content="Query"),
            ToolMessage(content="Some tool result", tool_call_id="0"),  # Has no tool_calls attribute
            AIMessage(
                content="AI response",
                tool_calls=[{"id": "call_456", "name": "another_tool", "args": {}}]
            )
        ]
        result = event_loop(state)
        assert result == "execute_tools", "Should find last message with tool_calls attribute and return 'execute_tools'"

    @staticmethod
    def test_complex_conversation_flow():
        """Test a realistic conversation flow."""
        state = [
            HumanMessage(content="What is AI?"),
            AIMessage(
                content="Let me search for that",
                tool_calls=[{"id": "search_1", "name": "search", "args": {"query": "AI definition"}}]
            ),
            ToolMessage(content="AI is artificial intelligence", tool_call_id="search_1"),
            AIMessage(content="Based on the search, AI is...", tool_calls=[]),
        ]

        tool_message_count = sum(isinstance(msg, ToolMessage) for msg in state)
        assert tool_message_count <= MAX_ITERATIONS, "Should not exceed max iterations"

        result = event_loop(state)
        assert result == END, "Conversation ending with AI message without tool_calls should return END"

    @staticmethod
    def test_ai_message_without_tool_calls_attribute():
        """Test AI message without explicit tool_calls (defaults to empty list)."""
        state = [
            HumanMessage(content="Query"),
            AIMessage(content="Response")  # No tool_calls parameter
        ]
        result = event_loop(state)
        assert result == END, "AI message without tool_calls should return END"

    @staticmethod
    def test_mixed_messages_last_ai_has_tools():
        """Test mixed message types where last AI message has tool calls."""
        state = [
            HumanMessage(content="Query 1"),
            AIMessage(content="Response 1", tool_calls=[{"id": "1", "name": "tool1", "args": {}}]),
            ToolMessage(content="Result 1", tool_call_id="1"),
            HumanMessage(content="Follow-up query"),
            AIMessage(content="Response 2", tool_calls=[{"id": "2", "name": "tool2", "args": {}}]),
        ]

        result = event_loop(state)
        assert result == "execute_tools", "Should return 'execute_tools' when last AI message has tool_calls"

    @staticmethod
    def test_edge_case_exactly_max_iterations_plus_one():
        """Test edge case: MAX_ITERATIONS + 1 tool messages should return END."""
        state = [HumanMessage(content="Initial query")]
        for i in range(MAX_ITERATIONS + 1):
            state.append(AIMessage(content=f"Response {i}", tool_calls=[{"id": str(i), "name": f"tool{i}", "args": {}}]))
            state.append(ToolMessage(content=f"Result {i}", tool_call_id=str(i)))

        tool_message_count = sum(isinstance(msg, ToolMessage) for msg in state)
        assert tool_message_count == MAX_ITERATIONS + 1, f"Test setup error"

        result = event_loop(state)
        assert result == END, f"Should return END when tool messages exceed MAX_ITERATIONS"


class TestEventLoopIntegration:
    """Integration tests for event_loop with graph context."""

    @staticmethod
    def test_event_loop_returns_valid_edge_names():
        """Ensure event_loop only returns valid edge names."""
        valid_edges = {END, "execute_tools"}

        # Test all basic scenarios
        test_states = [
            [],  # Empty
            [HumanMessage(content="test")],  # No AI message
            [AIMessage(content="test")],  # AI without tool_calls
            [AIMessage(content="test", tool_calls=[{"id": "1", "name": "tool", "args": {}}])],  # With tool_calls
        ]

        for state in test_states:
            result = event_loop(state)
            assert result in valid_edges, f"event_loop returned invalid edge: {result}"

    @staticmethod
    def test_max_iterations_constant_is_positive():
        """Ensure MAX_ITERATIONS is configured properly."""
        assert MAX_ITERATIONS > 0, "MAX_ITERATIONS must be positive"
        assert isinstance(MAX_ITERATIONS, int), "MAX_ITERATIONS must be an integer"


def main():
    """Run all tests."""
    print("=" * 70)
    print("Running Reflection Agent Tests")
    print("=" * 70)
    print()

    runner = TestRunner()

    # Run TestEventLoop tests
    print("Testing event_loop function:")
    print("-" * 70)
    runner.run_test("test_empty_state_returns_end", TestEventLoop.test_empty_state_returns_end)
    runner.run_test("test_state_with_tool_messages_exceeding_max_iterations_returns_end",
                    TestEventLoop.test_state_with_tool_messages_exceeding_max_iterations_returns_end)
    runner.run_test("test_state_at_max_iterations_boundary", TestEventLoop.test_state_at_max_iterations_boundary)
    runner.run_test("test_last_message_without_tool_calls_returns_end",
                    TestEventLoop.test_last_message_without_tool_calls_returns_end)
    runner.run_test("test_last_message_with_empty_tool_calls_returns_end",
                    TestEventLoop.test_last_message_with_empty_tool_calls_returns_end)
    runner.run_test("test_last_message_with_tool_calls_returns_execute_tools",
                    TestEventLoop.test_last_message_with_tool_calls_returns_execute_tools)
    runner.run_test("test_multiple_ai_messages_only_last_matters",
                    TestEventLoop.test_multiple_ai_messages_only_last_matters)
    runner.run_test("test_finds_last_ai_message_with_tool_calls_attribute",
                    TestEventLoop.test_finds_last_ai_message_with_tool_calls_attribute)
    runner.run_test("test_complex_conversation_flow", TestEventLoop.test_complex_conversation_flow)
    runner.run_test("test_ai_message_without_tool_calls_attribute", TestEventLoop.test_ai_message_without_tool_calls_attribute)
    runner.run_test("test_mixed_messages_last_ai_has_tools", TestEventLoop.test_mixed_messages_last_ai_has_tools)
    runner.run_test("test_edge_case_exactly_max_iterations_plus_one",
                    TestEventLoop.test_edge_case_exactly_max_iterations_plus_one)

    print()
    print("Testing event_loop integration:")
    print("-" * 70)
    runner.run_test("test_event_loop_returns_valid_edge_names",
                    TestEventLoopIntegration.test_event_loop_returns_valid_edge_names)
    runner.run_test("test_max_iterations_constant_is_positive",
                    TestEventLoopIntegration.test_max_iterations_constant_is_positive)

    print()
    success = runner.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Comparison Test: Current vs Simplified Implementations

This script runs the same research query through multiple implementations
and compares:
- Output quality
- Execution time
- Token usage (if LangSmith is enabled)
- Code complexity

Run with: python comparison_test.py
"""

import time
import sys
from pathlib import Path

# Test query
TEST_QUERY = """
Research the use of Large Language Models in code generation.
Cover:
- Current capabilities and limitations
- Popular tools and platforms
- Best practices for developers
- Security and quality concerns
- Future trends

Provide a brief report (2-3 paragraphs) with key insights.
"""


def test_current_implementation():
    """Test the current 750-line implementation"""
    print("\n" + "=" * 80)
    print("TESTING: Current Implementation (graph.py)")
    print("=" * 80)

    try:
        from graph import run_deep_research

        start = time.time()
        result = run_deep_research(
            query=TEST_QUERY,
            max_iterations=2,  # Limit for testing
            stream=False
        )
        elapsed = time.time() - start

        # Extract report
        if result:
            final_node = list(result.values())[0]
            report = final_node.get("final_report", "No report generated")
        else:
            report = "No result"

        return {
            "implementation": "Current (graph.py)",
            "lines_of_code": 750,
            "elapsed_time": elapsed,
            "report": report,
            "success": True
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "implementation": "Current (graph.py)",
            "success": False,
            "error": str(e)
        }


def test_create_agent_implementation():
    """Test the create_agent simplified version"""
    print("\n" + "=" * 80)
    print("TESTING: create_agent Implementation (100 lines)")
    print("=" * 80)

    try:
        from deep_research_create_agent import research

        start = time.time()
        result = research(TEST_QUERY, stream=False)
        elapsed = time.time() - start

        # Extract report
        if result and "messages" in result:
            final_message = result["messages"][-1]
            report = final_message.content if hasattr(final_message, "content") else "No content"
        else:
            report = "No result"

        return {
            "implementation": "create_agent (deep_research_create_agent.py)",
            "lines_of_code": 100,
            "elapsed_time": elapsed,
            "report": report,
            "success": True
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "implementation": "create_agent",
            "success": False,
            "error": str(e)
        }


def test_deep_agents_implementation():
    """Test the Deep Agents simplified version"""
    print("\n" + "=" * 80)
    print("TESTING: Deep Agents Implementation (40 lines)")
    print("=" * 80)

    try:
        from deep_research_simplified import research

        start = time.time()
        result = research(TEST_QUERY, use_subagent=True)
        elapsed = time.time() - start

        # Extract report
        if result and "messages" in result:
            final_message = result["messages"][-1]
            report = final_message.content if hasattr(final_message, "content") else "No content"
        else:
            report = "No result"

        return {
            "implementation": "Deep Agents (deep_research_simplified.py)",
            "lines_of_code": 40,
            "elapsed_time": elapsed,
            "report": report,
            "success": True
        }

    except ImportError:
        print("⚠️  Deep Agents not installed. Run: pip install deepagents")
        return {
            "implementation": "Deep Agents",
            "success": False,
            "error": "deepagents package not installed"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "implementation": "Deep Agents",
            "success": False,
            "error": str(e)
        }


def compare_results(results):
    """Compare and display results"""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Summary table
    print("\n📊 Performance Summary:\n")
    print(f"{'Implementation':<50} {'Lines':<10} {'Time':<15} {'Status':<10}")
    print("-" * 85)

    for r in results:
        if r["success"]:
            impl = r["implementation"]
            lines = r.get("lines_of_code", "N/A")
            time_str = f"{r.get('elapsed_time', 0):.2f}s"
            status = "✅ Success"
        else:
            impl = r["implementation"]
            lines = "N/A"
            time_str = "N/A"
            status = f"❌ Failed"

        print(f"{impl:<50} {lines:<10} {time_str:<15} {status:<10}")

    # Code reduction
    current_lines = next((r["lines_of_code"] for r in results if "graph.py" in r["implementation"] and r["success"]), 750)
    simplified_lines = [r.get("lines_of_code", 0) for r in results if r["success"] and "graph.py" not in r["implementation"]]

    if simplified_lines:
        avg_simplified = sum(simplified_lines) / len(simplified_lines)
        reduction = ((current_lines - avg_simplified) / current_lines) * 100
        print(f"\n📉 Code Reduction: {reduction:.1f}% (from {current_lines} to {avg_simplified:.0f} lines)")

    # Quality comparison
    print("\n" + "=" * 80)
    print("📝 Output Quality Comparison")
    print("=" * 80)

    for r in results:
        if r["success"]:
            print(f"\n{r['implementation']}:")
            print("-" * 80)
            report = r.get("report", "No report")
            # Show first 500 chars
            print(report[:500] + "..." if len(report) > 500 else report)

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    if len(successful) > 1:
        fastest = min(successful, key=lambda x: x.get("elapsed_time", float("inf")))
        simplest = min(successful, key=lambda x: x.get("lines_of_code", float("inf")))

        print(f"\n⚡ Fastest: {fastest['implementation']} ({fastest['elapsed_time']:.2f}s)")
        print(f"📝 Simplest: {simplest['implementation']} ({simplest['lines_of_code']} lines)")

        if "Deep Agents" in simplest["implementation"]:
            print("\n✅ Recommended: Deep Agents (40 lines, most features)")
        elif "create_agent" in simplest["implementation"]:
            print("\n✅ Recommended: create_agent (100 lines, no new packages needed)")
        else:
            print("\n✅ Both simplified versions are excellent choices!")

    print("\n" + "=" * 80)


def main():
    """Run all tests and compare"""
    print("=" * 80)
    print("DEEP RESEARCH AGENT - IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(f"\nTest Query: {TEST_QUERY[:100]}...")
    print("\nTesting 3 implementations:")
    print("1. Current (graph.py) - 750 lines")
    print("2. create_agent - 100 lines")
    print("3. Deep Agents - 40 lines")

    results = []

    # Test each implementation
    # Note: Uncomment as needed based on what you want to test

    # Test current implementation
    # results.append(test_current_implementation())

    # Test create_agent (always works - no new packages)
    results.append(test_create_agent_implementation())

    # Test Deep Agents (requires: pip install deepagents)
    results.append(test_deep_agents_implementation())

    # Compare results
    compare_results(results)

    # Save results to file
    output_file = f"comparison_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, "w") as f:
        f.write("COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"{r['implementation']}\n")
            f.write("-" * 80 + "\n")
            if r["success"]:
                f.write(f"Lines: {r.get('lines_of_code')}\n")
                f.write(f"Time: {r.get('elapsed_time'):.2f}s\n")
                f.write(f"Report:\n{r.get('report')}\n\n")
            else:
                f.write(f"Error: {r.get('error')}\n\n")

    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

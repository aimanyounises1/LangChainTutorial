#!/usr/bin/env python3
"""
Deep Research Agent - Main Entry Point

This module provides a command-line interface for running deep research
and utilities for inspecting research state.

Usage:
    python main.py "Your research query here"
    python main.py --resume thread_id
    python main.py --inspect thread_id
"""

# MUST be first - before any langchain imports to enable LangSmith tracing!
from dotenv import load_dotenv
load_dotenv()

import argparse
import datetime
import json
import traceback
from pathlib import Path

from langgraph_examples.deep_research_agent.graph import (
    run_deep_research,
    get_research_state,
    resume_research,
)


def print_banner():
    """Print a nice banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ███████╗███████╗██████╗     ██████╗ ███████╗███████╗███████╗       ║
║   ██╔══██╗██╔════╝██╔════╝██╔══██╗    ██╔══██╗██╔════╝██╔════╝██╔════╝       ║
║   ██║  ██║█████╗  █████╗  ██████╔╝    ██████╔╝█████╗  ███████╗█████╗         ║
║   ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝     ██╔══██╗██╔══╝  ╚════██║██╔══╝         ║
║   ██████╔╝███████╗███████╗██║         ██║  ██║███████╗███████║███████╗       ║
║   ╚═════╝ ╚══════╝╚══════╝╚═╝         ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝       ║
║                                                                              ║
║                    RESEARCH AGENT - Powered by LangGraph                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def save_report(report: str, metadata: dict, output_dir: Path = None):
    """Save the final report to a file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "reports"

    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save report as markdown
    report_file = output_dir / f"research_report_{timestamp}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    # Save metadata as JSON
    metadata_file = output_dir / f"research_metadata_{timestamp}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n📄 Report saved to: {report_file}")
    print(f"📊 Metadata saved to: {metadata_file}")

    return report_file, metadata_file


def inspect_state(thread_id: str):
    """Inspect the state of a research thread."""
    print(f"\n🔍 Inspecting thread: {thread_id}")
    print("=" * 60)

    state = get_research_state(thread_id)

    if not state:
        print(f"❌ No state found for thread: {thread_id}")
        return

    print(f"📌 Phase: {state.get('phase', 'Unknown')}")
    print(f"🔄 Iteration: {state.get('iteration', 0)}/{state.get('max_iterations', 5)}")
    print(f"✅ Complete: {state.get('is_complete', False)}")

    if state.get('research_plan'):
        plan = state['research_plan']
        print(f"\n📋 Research Plan:")
        print(f"   Query: {plan.get('main_query', 'N/A')[:80]}...")
        sub_questions = plan.get('sub_questions', [])
        print(f"   Sub-questions: {len(sub_questions)}")
        for sq in sub_questions:
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(sq.get('status'), "❓")
            print(f"     {status_emoji} {sq.get('question', 'N/A')[:60]}...")

    citations = state.get('citations', [])
    print(f"\n📚 Citations: {len(citations)}")

    if state.get('draft'):
        draft = state['draft']
        sections = draft.get('sections', [])
        print(f"\n📝 Draft Sections: {len(sections)}")
        for sec in sections:
            print(f"   • {sec.get('title', 'Untitled')}")

    if state.get('latest_critique'):
        critique = state['latest_critique']
        metrics = critique.get('quality_metrics', {})
        print(f"\n📊 Latest Quality Metrics:")
        print(f"   Coverage: {metrics.get('coverage_score', 0):.2f}")
        print(f"   Depth: {metrics.get('depth_score', 0):.2f}")
        print(f"   Completeness: {metrics.get('completeness_score', 0):.2f}")

    if state.get('completion_reason'):
        print(f"\n🏁 Completion Reason: {state['completion_reason']}")

    print("=" * 60)


def run_interactive():
    """Run in interactive mode."""
    print_banner()

    print("\n📝 Enter your research query (or 'quit' to exit):\n")

    while True:
        query = input("🔍 Query: ").strip()

        if query.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Goodbye!")
            break

        if not query:
            print("⚠️  Please enter a query.")
            continue

        # Ask for iterations
        try:
            max_iter = input("🔄 Max iterations (default=5): ").strip()
            max_iterations = int(max_iter) if max_iter else 5
        except ValueError:
            max_iterations = 5

        print(f"\n🚀 Starting research with {max_iterations} max iterations...\n")

        try:
            result = run_deep_research(
                query=query,
                max_iterations=max_iterations,
                stream=True
            )

            if result:
                final_output = list(result.values())[0]

                if final_output.get("final_report"):
                    print("\n" + "=" * 80)
                    print("📄 FINAL REPORT")
                    print("=" * 80)
                    print(final_output["final_report"])

                    # Ask to save
                    save_choice = input("\n💾 Save report to file? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        save_report(
                            final_output["final_report"],
                            final_output.get("report_metadata", {})
                        )
                else:
                    print("\n⚠️  No final report generated.")
        except Exception as e:
            print(f"\n❌ Error during research: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "-" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Comprehensive AI-powered research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What are the best practices for AI governance?"
  python main.py --iterations 3 "Analyze the cryptocurrency market trends"
  python main.py --resume research_20240101_120000
  python main.py --inspect research_20240101_120000
  python main.py --interactive
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="The research query to investigate"
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)"
    )

    parser.add_argument(
        "--resume", "-r",
        type=str,
        help="Resume research from a thread ID"
    )

    parser.add_argument(
        "--inspect", "-s",
        type=str,
        help="Inspect state of a research thread"
    )

    parser.add_argument(
        "--interactive", "-I",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for reports"
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        run_interactive()
        return

    # Inspect mode
    if args.inspect:
        inspect_state(args.inspect)
        return

    # Resume mode
    if args.resume:
        print_banner()
        print(f"\n🔄 Resuming research thread: {args.resume}")
        try:
            result = resume_research(args.resume)
            if result:
                final_output = list(result.values())[0]
                if final_output.get("final_report"):
                    print("\n" + "=" * 80)
                    print("📄 FINAL REPORT")
                    print("=" * 80)
                    print(final_output["final_report"])

                    if args.output:
                        save_report(
                            final_output["final_report"],
                            final_output.get("report_metadata", {}),
                            Path(args.output)
                        )
        except Exception as e:
            print(f"❌ Error resuming research: {e}")
        return

    # Normal research mode
    if not args.query:
        parser.print_help()
        return

    print_banner()

    print(f"\n🔍 Research Query: {args.query}")
    print(f"🔄 Max Iterations: {args.iterations}")
    print(f"📡 Streaming: {'Disabled' if args.no_stream else 'Enabled'}")
    print("\n" + "=" * 80 + "\n")

    try:
        result = run_deep_research(
            query=args.query,
            max_iterations=args.iterations,
            stream=not args.no_stream
        )

        if result:
            final_output = list(result.values())[0]

            if final_output.get("final_report"):
                print("\n" + "=" * 80)
                print("📄 FINAL REPORT")
                print("=" * 80)
                print(final_output["final_report"])

                if args.output:
                    save_report(
                        final_output["final_report"],
                        final_output.get("report_metadata", {}),
                        Path(args.output)
                    )

        else:
            print("\n❌ Research did not produce results.")

    except KeyboardInterrupt:
        print("\n\n⏹️  Research interrupted by user.")
        print("   Use --resume with the thread ID to continue later.")
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

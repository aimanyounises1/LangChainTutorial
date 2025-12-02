import datetime
import json
import operator
import sqlite3
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any

# Load environment variables BEFORE any langchain imports to enable LangSmith tracing!
from dotenv import load_dotenv
load_dotenv(verbose=True)

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

from langgraph_examples.deep_research_agent.agents import (
    # Planner
    planner_chain,
    planner_parser,
    create_default_plan,
    validate_research_plan,
    # Researcher
    research_sub_question,
    # Synthesizer
    synthesize_findings,
    update_draft_with_section,
    initialize_draft,
    # Critic
    critique_draft,
    # Report Generator
    generate_final_report,
    format_report_as_markdown,
    calculate_report_statistics,
)
from langgraph_examples.deep_research_agent.schemas import (
    ResearchPhase,
    ResearchPlan,
    ResearchDraft,
    Citation,
    CritiqueResult,
    QualityMetrics,
    StopConditionConfig,
)
from langgraph_examples.deep_research_agent.text_parser import (
    parse_text_tool_calls,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(CHECKPOINTS_DIR / "deep_research_state.db")
# Use direct sqlite3 connection for persistent file-based storage
# (from_conn_string returns a context manager, not a direct instance)
_checkpoint_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
checkpointer = SqliteSaver(_checkpoint_conn)

DEFAULT_STOP_CONFIG = StopConditionConfig(
    min_coverage_score=0.7,
    min_depth_score=0.6,
    min_citation_density=0.5,
    min_completeness_score=0.7,
    max_iterations=5,
    max_consecutive_no_improvement=2,
    min_sub_questions_completed=0.8
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class DeepResearchGraphState(TypedDict, total=False):
    """
    State for the deep research graph using TypedDict for LangGraph compatibility.

    Using total=False makes all fields optional at the TypedDict level,
    which aligns with LangGraph's pattern of partial state updates.
    """
    # Input
    original_query: str

    # Research plan (serialized ResearchPlan)
    research_plan: Dict[str, Any]

    # Draft evolution (serialized ResearchDraft)
    draft: Dict[str, Any]

    # Citations collection (list of serialized Citations, uses reducer)
    citations: Annotated[List[Dict[str, Any]], operator.add]

    # Progress tracking
    phase: str
    current_sub_question_index: int
    iteration: int
    max_iterations: int

    # Quality tracking (critique_history uses reducer)
    critique_history: Annotated[List[Dict[str, Any]], operator.add]
    latest_critique: Dict[str, Any]

    # Search results (for current iteration)
    current_search_results: str

    # Completion
    is_complete: bool
    completion_reason: str

    # Final output
    final_report: str
    report_metadata: Dict[str, Any]

    # Messages (for debugging/logging, uses reducer)
    messages: Annotated[List[BaseMessage], operator.add]


def serialize_plan(plan: ResearchPlan) -> Dict[str, Any]:
    """Serialize ResearchPlan to dict."""
    return plan.model_dump()


def deserialize_plan(data: Dict[str, Any]) -> ResearchPlan:
    """Deserialize dict to ResearchPlan."""
    return ResearchPlan(**data)


def serialize_draft(draft: ResearchDraft) -> Dict[str, Any]:
    """Serialize ResearchDraft to dict."""
    return draft.model_dump()


def deserialize_draft(data: Dict[str, Any]) -> ResearchDraft:
    """Deserialize dict to ResearchDraft."""
    return ResearchDraft(**data)


def serialize_citation(citation: Citation) -> Dict[str, Any]:
    """Serialize Citation to dict."""
    return citation.model_dump()


def deserialize_citation(data: Dict[str, Any]) -> Citation:
    """Deserialize dict to Citation."""
    return Citation(**data)


def serialize_critique(critique: CritiqueResult) -> Dict[str, Any]:
    """Serialize CritiqueResult to dict."""
    return critique.model_dump()


def deserialize_critique(data: Dict[str, Any]) -> CritiqueResult:
    """Deserialize dict to CritiqueResult."""
    return CritiqueResult(**data)


def extract_query_from_state(state: DeepResearchGraphState) -> str:
    def extract_text_from_content(content) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multi-modal format: [{'type': 'text', 'text': '...'}]
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            return " ".join(text_parts).strip()
        return str(content) if content else ""

    # First, try to get from original_query
    original_query = state.get("original_query")
    if original_query:
        extracted = extract_text_from_content(original_query)
        if extracted.strip():
            return extracted

    # Fallback: extract from messages (LangGraph Studio format)
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            extracted = extract_text_from_content(msg.content)
            if extracted.strip():
                return extracted
        # Handle dict format (from JSON input)
        if isinstance(msg, dict) and msg.get("type") == "human":
            content = msg.get("content", "")
            extracted = extract_text_from_content(content)
            if extracted.strip():
                return extracted

    raise ValueError("No query found in state. Provide 'original_query' or 'messages' with a HumanMessage.")


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def planning_node(state: DeepResearchGraphState) -> Dict[str, Any]:
    """
    Planning node - Creates the research plan from the original query.
    
    Handles input from both direct invocation and LangGraph Studio.
    """
    print(f"\n{'=' * 60}")
    print("[PLANNER] Creating research plan...")
    print(f"{'=' * 60}")

    # Extract query from state (handles both input formats)
    query = extract_query_from_state(state)
    print(f"[PLANNER] Query: {query[:100]}...")

    # Store original_query in state for downstream nodes
    updates = {"original_query": query}

    try:
        # Invoke planner chain
        result = planner_chain.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # Parse text-based tool calls if needed
        if not result.tool_calls:
            result = parse_text_tool_calls(result)

        if result.tool_calls:
            parsed = planner_parser.invoke(result)
            if parsed:
                plan = parsed.research_plan
                is_valid, issues = validate_research_plan(plan)

                if is_valid:
                    print(f"[PLANNER] Created plan with {len(plan.sub_questions)} sub-questions")
                    for sq in plan.sub_questions:
                        print(f"  [{sq.priority}] {sq.question}")

                    return {
                        "original_query": query,  # Store for downstream nodes
                        "research_plan": serialize_plan(plan),
                        "phase": ResearchPhase.RESEARCHING.value,
                        "current_sub_question_index": 0,
                        "messages": [
                            AIMessage(content=f"Created research plan with {len(plan.sub_questions)} sub-questions")]
                    }
                else:
                    print(f"[PLANNER] Plan validation failed: {issues}")
    except Exception as e:
        print(f"[PLANNER] Error: {e}")

    # Fallback to default plan
    print("[PLANNER] Using fallback plan")
    default_plan = create_default_plan(query)

    return {
        "original_query": query,  # Store for downstream nodes
        "research_plan": serialize_plan(default_plan),
        "phase": ResearchPhase.RESEARCHING.value,
        "current_sub_question_index": 0,
        "messages": [
            AIMessage(content=f"Created fallback research plan with {len(default_plan.sub_questions)} sub-questions")]
    }


def research_node(state: DeepResearchGraphState) -> Dict[str, Any]:
    """
    Research node - Researches the current sub-question.
    """
    print(f"\n{'=' * 60}")
    print("[RESEARCHER] Gathering information...")
    print(f"{'=' * 60}")

    plan_data = state["research_plan"]
    plan = deserialize_plan(plan_data)

    pending_questions = [(i, sq) for i, sq in enumerate(plan.sub_questions)
                         if sq.status in ("pending", "in_progress")]

    if not pending_questions:
        print("[RESEARCHER] No pending sub-questions")
        return {
            "phase": ResearchPhase.CRITIQUING.value,
            "current_search_results": "",
            "messages": [AIMessage(content="No more sub-questions to research")]
        }

    # Get current sub-question
    idx, sub_question = pending_questions[0]
    sub_question.status = "in_progress"

    print(f"[RESEARCHER] Researching: {sub_question.question}")

    # Get existing citations count
    existing_citations = len(state.get("citations", []))

    # Get previous findings for context
    draft_data = state.get("draft")
    previous_findings = ""
    if draft_data:
        draft = deserialize_draft(draft_data)
        previous_findings = "\n".join(s.content[:500] for s in draft.sections)

    # Execute research
    content, new_citations, queries_used = research_sub_question(
        sub_question=sub_question,
        main_query=plan.main_query,
        previous_findings=previous_findings,
        existing_citations=existing_citations
    )

    # Update sub-question with queries used
    sub_question.search_queries = queries_used

    # Serialize new citations
    serialized_citations = [serialize_citation(c) for c in new_citations]

    print(f"[RESEARCHER] Found {len(new_citations)} new sources")

    # Update plan with sub-question status
    plan.sub_questions[idx] = sub_question

    return {
        "research_plan": serialize_plan(plan),
        "current_sub_question_index": idx,
        "current_search_results": content,
        "citations": serialized_citations,
        "phase": ResearchPhase.SYNTHESIZING.value,
        "messages": [AIMessage(content=f"Researched: {sub_question.question}, found {len(new_citations)} sources")]
    }


def synthesize_node(state: DeepResearchGraphState) -> Dict[str, Any]:
    """
    Synthesize node - Integrates findings into the draft.
    """
    print(f"\n{'=' * 60}")
    print("[SYNTHESIZER] Integrating findings...")
    print(f"{'=' * 60}")

    plan_data = state["research_plan"]
    plan = deserialize_plan(plan_data)

    # Get current sub-question
    idx = state["current_sub_question_index"]
    sub_question = plan.sub_questions[idx]

    # Get current draft
    draft_data = state.get("draft")
    current_draft = deserialize_draft(draft_data) if draft_data else None

    # Get all citations
    all_citations = [deserialize_citation(c) for c in state.get("citations", [])]

    # Get search results
    search_results = state.get("current_search_results", "")

    if not search_results:
        print("[SYNTHESIZER] No search results to synthesize")
        return {
            "phase": ResearchPhase.CRITIQUING.value,
            "messages": [AIMessage(content="No search results to synthesize")]
        }

    # Synthesize findings
    updated_section, new_citations, notes = synthesize_findings(
        sub_question=sub_question,
        search_results=search_results,
        main_query=plan.main_query,
        current_draft=current_draft,
        available_citations=all_citations,
        expected_sections=plan.expected_sections
    )

    # Update draft
    if current_draft is None:
        current_draft = initialize_draft(plan.main_query)

    current_draft = update_draft_with_section(current_draft, updated_section, plan.main_query)

    # Mark sub-question as completed
    sub_question.status = "completed"
    sub_question.findings = updated_section.content[:500]
    sub_question.citations = updated_section.citations
    plan.sub_questions[idx] = sub_question

    print(f"[SYNTHESIZER] Updated section: {updated_section.title}")
    print(f"[SYNTHESIZER] Draft now has {len(current_draft.sections)} sections")

    # Serialize new citations if any
    serialized_new_citations = [serialize_citation(c) for c in new_citations]

    return {
        "research_plan": serialize_plan(plan),
        "draft": serialize_draft(current_draft),
        "citations": serialized_new_citations,
        "phase": ResearchPhase.CRITIQUING.value,
        "messages": [AIMessage(content=f"Synthesized findings into: {updated_section.title}")]
    }


def critique_node(state: DeepResearchGraphState) -> Dict[str, Any]:
    """
    Critique node - Evaluates the draft and determines next action.
    """
    print(f"\n{'=' * 60}")
    print("[CRITIC] Evaluating research quality...")
    print(f"{'=' * 60}")

    plan_data = state["research_plan"]
    plan = deserialize_plan(plan_data)

    draft_data = state.get("draft")
    draft = deserialize_draft(draft_data) if draft_data else None

    all_citations = [deserialize_citation(c) for c in state.get("citations", [])]

    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", DEFAULT_STOP_CONFIG.max_iterations)

    # Get critique history
    critique_history = [deserialize_critique(c) for c in state.get("critique_history", [])]

    # Perform critique
    critique, next_action = critique_draft(
        draft=draft,
        plan=plan,
        citations=all_citations,
        iteration=iteration,
        max_iterations=max_iterations,
        critique_history=critique_history,
        stop_config=DEFAULT_STOP_CONFIG
    )

    # Log critique results
    metrics = critique.quality_metrics
    print(f"[CRITIC] Iteration {iteration}/{max_iterations}")
    print(f"[CRITIC] Coverage: {metrics.coverage_score:.2f}")
    print(f"[CRITIC] Depth: {metrics.depth_score:.2f}")
    print(f"[CRITIC] Completeness: {metrics.completeness_score:.2f}")
    print(f"[CRITIC] Recommended action: {next_action}")

    # Determine next phase
    if next_action == "finalize" or next_action == "stop":
        next_phase = ResearchPhase.FINALIZING.value
        is_complete = True
        completion_reason = critique.reasoning
    else:
        next_phase = ResearchPhase.RESEARCHING.value
        is_complete = False
        completion_reason = None

    return {
        "latest_critique": serialize_critique(critique),
        "critique_history": [serialize_critique(critique)],
        "iteration": iteration,
        "phase": next_phase,
        "is_complete": is_complete,
        "completion_reason": completion_reason,
        "messages": [AIMessage(content=f"Critique: {next_action} - {critique.reasoning[:100]}")]
    }


def finalize_node(state: DeepResearchGraphState) -> Dict[str, Any]:
    """
    Finalize node - Generates the final polished report.
    """
    print(f"\n{'=' * 60}")
    print("[REPORT GENERATOR] Creating final report...")
    print(f"{'=' * 60}")

    plan_data = state["research_plan"]
    plan = deserialize_plan(plan_data)

    draft_data = state.get("draft")
    if not draft_data:
        return {
            "final_report": "Error: No draft available to finalize",
            "report_metadata": {"error": True},
            "phase": ResearchPhase.COMPLETE.value,
            "messages": [AIMessage(content="Error: No draft to finalize")]
        }

    draft = deserialize_draft(draft_data)
    all_citations = [deserialize_citation(c) for c in state.get("citations", [])]

    # Get quality metrics from latest critique
    critique_data = state.get("latest_critique")
    if critique_data:
        critique = deserialize_critique(critique_data)
        quality_metrics = critique.quality_metrics
    else:
        quality_metrics = QualityMetrics(
            coverage_score=0.5,
            depth_score=0.5,
            citation_density=0.5,
            coherence_score=0.5,
            completeness_score=0.5
        )

    # Generate final report
    final_report, metadata = generate_final_report(
        draft=draft,
        plan=plan,
        citations=all_citations,
        quality_metrics=quality_metrics
    )

    # Format and calculate statistics
    final_report = format_report_as_markdown(final_report)
    stats = calculate_report_statistics(final_report, all_citations)

    metadata.update(stats)
    metadata["completion_reason"] = state.get("completion_reason", "Unknown")
    metadata["iterations"] = state.get("iteration", 0)

    print(f"[REPORT GENERATOR] Report generated:")
    print(f"  - Words: {stats['word_count']}")
    print(f"  - Sections: {stats['section_count']}")
    print(f"  - Citations: {stats['citation_references']}")

    return {
        "final_report": final_report,
        "report_metadata": metadata,
        "phase": ResearchPhase.COMPLETE.value,
        "is_complete": True,
        "messages": [AIMessage(
            content=f"Final report generated: {stats['word_count']} words, {stats['section_count']} sections")]
    }


def route_after_critique(state: DeepResearchGraphState) -> str:
    """
    Determine the next node after critique.
    """
    phase = state.get("phase", "")
    is_complete = state.get("is_complete", False)

    if is_complete or phase == ResearchPhase.FINALIZING.value:
        return "finalize"
    else:
        return "research"


def check_if_done(state: DeepResearchGraphState) -> str:
    """
    Check if the research is complete.
    """
    phase = state.get("phase", "")

    if phase == ResearchPhase.COMPLETE.value:
        return END

    return "continue"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def build_deep_research_graph(checkpointer=None):
    """Build the deep research state graph."""

    builder = StateGraph(DeepResearchGraphState)

    # Add nodes
    builder.add_node("plan", planning_node)
    builder.add_node("research", research_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critique", critique_node)
    builder.add_node("finalize", finalize_node)

    # Add edges
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")
    builder.add_edge("synthesize", "critique")

    # Conditional edge after critique
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "research": "research",
            "finalize": "finalize"
        }
    )

    builder.add_edge("finalize", END)
    builder.set_entry_point("plan")

    if checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
    else:
        graph = builder.compile()

    return graph


# Build the graph
deep_research_graph = build_deep_research_graph(checkpointer=checkpointer)

# Visualize
try:
    print("\n" + deep_research_graph.get_graph().draw_ascii())
    deep_research_graph.get_graph().draw_mermaid_png(
        output_file_path=str(Path(__file__).parent / "deep_research_graph.png")
    )
except Exception as e:
    print(f"Could not generate graph visualization: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_deep_research(
        query: str,
        thread_id: str | None = None,
        max_iterations: int = 5,
        stream: bool = True
) -> Dict[str, Any]:
    """
    Run the deep research agent on a query.
    
    Args:
        query: The research query
        thread_id: Optional thread ID for checkpointing
        max_iterations: Maximum research iterations
        stream: Whether to stream progress updates
    
    Returns:
        Final state with report
    """
    if thread_id is None:
        thread_id = f"research_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = RunnableConfig(configurable={"thread_id": thread_id})

    initial_state = {
        "original_query": query,
        "research_plan": None,
        "draft": None,
        "citations": [],
        "phase": ResearchPhase.PLANNING.value,
        "current_sub_question_index": 0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "critique_history": [],
        "latest_critique": None,
        "current_search_results": "",
        "is_complete": False,
        "completion_reason": None,
        "final_report": None,
        "report_metadata": None,
        "messages": [HumanMessage(content=query)]
    }

    print(f"\n{'=' * 80}")
    print(f"DEEP RESEARCH AGENT")
    print(f"{'=' * 80}")
    print(f"Query: {query}")
    print(f"Thread ID: {thread_id}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'=' * 80}\n")

    if stream:
        final_state = None
        for i, chunk in enumerate(deep_research_graph.stream(initial_state, config)):
            node_name = list(chunk.keys())[0]
            update = chunk[node_name]

            phase = update.get("phase", "")
            print(f"\n[Step {i}] Node: {node_name} | Phase: {phase}")

            if update.get("messages"):
                for msg in update["messages"]:
                    if isinstance(msg, AIMessage):
                        print(f"  → {msg.content[:100]}...")

            final_state = chunk

        return final_state
    else:
        return deep_research_graph.invoke(initial_state, config=config)


def get_research_state(thread_id: str) -> Dict[str, Any] | None:
    """Get the current state for a research thread."""
    config = RunnableConfig(configurable={"thread_id": thread_id})
    state = deep_research_graph.get_state(config)
    return state.values if state else None


def resume_research(thread_id: str) -> Dict[str, Any]:
    """Resume a paused or interrupted research session."""
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})
    state = deep_research_graph.get_state(config)

    if not state or not state.values:
        raise ValueError(f"No state found for thread: {thread_id}")

    print(f"Resuming research from thread: {thread_id}")
    print(f"Current phase: {state.values.get('phase')}")
    print(f"Iteration: {state.values.get('iteration')}")

    # Continue from current state
    final_state = None
    for chunk in deep_research_graph.stream(None, config):
        node_name = list(chunk.keys())[0]
        print(f"Resumed at node: {node_name}")
        final_state = chunk

    return final_state


if __name__ == "__main__":
    # Test the deep research agent
    test_query = """
    Research AI-Powered Security Operations Centers (SOC). 
    I want to understand the current landscape, key players, 
    startup funding, challenges, and future trends.
    """

    result = run_deep_research(
        query=test_query,
        max_iterations=3,
        stream=True
    )

    if result:
        # Get the final node's output
        final_output = list(result.values())[0]

        if final_output.get("final_report"):
            print("\n" + "=" * 80)
            print("FINAL REPORT")
            print("=" * 80)
            print(final_output["final_report"][:3000])
            print("\n... (truncated)")

            if final_output.get("report_metadata"):
                print("\n" + "=" * 80)
                print("REPORT METADATA")
                print("=" * 80)
                print(json.dumps(final_output["report_metadata"], indent=2))

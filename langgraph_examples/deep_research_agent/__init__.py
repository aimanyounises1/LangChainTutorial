"""
Deep Research Agent Package

A comprehensive multi-agent system for conducting deep research on any topic.
The system uses multiple specialized sub-agents:

1. **Planner**: Analyzes queries and creates research plans
2. **Researcher**: Gathers information through targeted searches
3. **Synthesizer**: Integrates findings into evolving drafts
4. **Critic**: Evaluates quality and determines completion
5. **Report Generator**: Creates polished final reports

## Usage:

```python
from langgraph_examples.deep_research_agent import run_deep_research

# Run research on a topic
result = run_deep_research(
    query="What are the best practices for implementing AI in healthcare?",
    max_iterations=5
)

# Access the final report
print(result["final_report"])
```

## Features:

- Iterative research with quality feedback loops
- Citation management with proper attribution
- SQLite checkpointing for recovery
- Configurable stop conditions
- Professional report formatting
"""

from langgraph_examples.deep_research_agent.graph import (
    deep_research_graph,
    run_deep_research,
    get_research_state,
    resume_research,
    build_deep_research_graph,
    CHECKPOINTS_DIR,
    CHECKPOINT_DB,
)

from langgraph_examples.deep_research_agent.schemas import (
    ResearchPhase,
    ResearchPlan,
    ResearchDraft,
    Citation,
    SubQuestion,
    CritiqueResult,
    QualityMetrics,
    StopConditionConfig,
    DraftSection,
    DeepResearchState,
)

__all__ = [
    # Main graph and functions
    "deep_research_graph",
    "run_deep_research",
    "get_research_state",
    "resume_research",
    "build_deep_research_graph",
    "CHECKPOINTS_DIR",
    "CHECKPOINT_DB",
    # Schemas
    "ResearchPhase",
    "ResearchPlan",
    "ResearchDraft",
    "Citation",
    "SubQuestion",
    "CritiqueResult",
    "QualityMetrics",
    "StopConditionConfig",
    "DraftSection",
    "DeepResearchState",
]

__version__ = "1.0.0"

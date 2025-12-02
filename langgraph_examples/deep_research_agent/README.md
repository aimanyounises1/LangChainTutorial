# Deep Research Agent 🔬

A comprehensive multi-agent system for conducting deep, iterative research on any topic. Built with LangGraph, this system employs specialized sub-agents that work together to produce professional-quality research reports.

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   PLANNER   │ ──► Creates research plan
                    └──────┬──────┘     with sub-questions
                           │
                           ▼
              ┌───────────────────────┐
              │                       │
              ▼                       │
        ┌───────────┐                 │
        │ RESEARCHER │ ──► Searches   │
        └─────┬─────┘     for info    │
              │                       │
              ▼                       │
       ┌────────────┐                 │
       │ SYNTHESIZER │ ──► Builds     │
       └──────┬─────┘     draft       │
              │                       │
              ▼                       │
        ┌──────────┐                  │
        │  CRITIC  │ ──► Evaluates ───┘
        └────┬─────┘     quality      (loop if not complete)
             │
             │ (if complete)
             ▼
     ┌───────────────┐
     │ REPORT GEN    │ ──► Polishes
     └───────┬───────┘     report
             │
             ▼
        ┌─────────┐
        │   END   │
        └─────────┘
```

## 🤖 Sub-Agents

### 1. Planner Agent
- Analyzes the research query
- Breaks it down into 4-7 specific sub-questions
- Creates a structured research plan
- Defines methodology and expected sections

### 2. Researcher Agent
- Generates optimal search queries
- Executes web searches via Tavily
- Collects and structures sources
- Manages citation creation

### 3. Synthesizer Agent
- Analyzes new search results
- Extracts key insights
- Integrates findings into evolving draft
- Applies inline citations

### 4. Critic Agent
- Evaluates draft quality
- Scores: coverage, depth, citations, coherence, completeness
- Identifies gaps and improvements
- Determines when research is complete

### 5. Report Generator Agent
- Creates polished final report
- Formats citations properly
- Generates executive summary
- Ensures professional structure

## 📊 Quality Metrics

The system uses multiple quality metrics to determine research completeness:

| Metric | Description | Target |
|--------|-------------|--------|
| Coverage Score | How well sub-questions are addressed | ≥ 0.7 |
| Depth Score | Analytical depth of content | ≥ 0.6 |
| Citation Density | Claims with proper citations | ≥ 0.5 |
| Coherence Score | Logical flow and structure | ≥ 0.7 |
| Completeness Score | Overall readiness | ≥ 0.7 |

## 🚀 Usage

### Command Line

```bash
# Basic research
python main.py "What are the best practices for AI governance?"

# With custom iterations
python main.py --iterations 3 "Analyze cryptocurrency market trends"

# Interactive mode
python main.py --interactive

# Resume interrupted research
python main.py --resume research_20240101_120000

# Inspect research state
python main.py --inspect research_20240101_120000
```

### Programmatic Usage

```python
from langgraph_examples.deep_research_agent import run_deep_research

# Run research
result = run_deep_research(
    query="Research AI-Powered Security Operations Centers",
    max_iterations=5,
    stream=True
)

# Access the final report
final_output = list(result.values())[0]
print(final_output["final_report"])
print(final_output["report_metadata"])
```

### LangGraph Studio

```bash
# From the deep_research_agent directory
langgraph up
```

## 🔄 Stop Conditions

The research loop stops when any of these conditions are met:

1. **Quality Threshold Met**: All quality scores meet minimum requirements
2. **Max Iterations Reached**: Default 5 iterations
3. **No Improvement**: 2 consecutive iterations with no quality improvement
4. **Sub-questions Completed**: ≥80% of sub-questions fully addressed

## 📁 Project Structure

```
deep_research_agent/
├── __init__.py          # Package exports
├── schemas.py           # Pydantic models & state definitions
├── graph.py             # Main LangGraph orchestration
├── main.py              # CLI entry point
├── text_parser.py       # Text-based tool call parsing
├── langgraph.json       # LangGraph Studio config
├── agents/
│   ├── __init__.py      # Agent exports
│   ├── planner.py       # Planning agent
│   ├── researcher.py    # Research agent
│   ├── synthesizer.py   # Synthesis agent
│   ├── critic.py        # Critique agent
│   └── report_generator.py  # Report generation
├── checkpoints/         # SQLite state persistence
└── reports/             # Generated report outputs
```

## 🛠️ Configuration

### Stop Condition Configuration

```python
from langgraph_examples.deep_research_agent import StopConditionConfig

config = StopConditionConfig(
    min_coverage_score=0.7,
    min_depth_score=0.6,
    min_citation_density=0.5,
    min_completeness_score=0.7,
    max_iterations=5,
    max_consecutive_no_improvement=2,
    min_sub_questions_completed=0.8
)
```

### LLM Configuration

By default, the system uses Ollama with `qwen3:30b-a3b`. To change:

```python
# In each agent file, modify the create_*_llm function
llm = ChatOllama(
    model='your-model-name',
    temperature=0,
    num_ctx=8192,
)
```

## 📝 Output Format

The final report follows this structure:

```markdown
# Research Report: [Topic]

## Executive Summary
[150-200 word summary]

## 1. Introduction
[Background and context]

## 2. Methodology
[Research approach]

## 3. Key Findings
[Main discoveries with citations]

## 4. Analysis & Discussion
[Deeper analysis]

## 5. Challenges & Considerations
[Limitations and risks]

## 6. Future Outlook
[Trends and predictions]

## 7. Conclusion
[Summary and recommendations]

## References
[Full citation list]
```

## 🔧 Dependencies

- `langgraph`
- `langchain-core`
- `langchain-ollama`
- `langchain-tavily`
- `pydantic`
- `python-dotenv`

## 📜 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit PRs for improvements or bug fixes.

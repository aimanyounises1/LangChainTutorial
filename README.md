<p align="center">
  <img src="https://img.shields.io/badge/LangChain-2.0+-blue?style=for-the-badge&logo=chainlink" alt="LangChain"/>
  <img src="https://img.shields.io/badge/LangGraph-0.3+-green?style=for-the-badge&logo=graphql" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Python-3.12+-yellow?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">LangChain & LangGraph Tutorial</h1>

<p align="center">
  <strong>A hands-on tutorial for building intelligent AI agents with LangChain and LangGraph</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## What You'll Learn

| Pattern | Description | Location |
|---------|-------------|----------|
| **ReAct Agent** | Reasoning + Acting loop with tool use | `langgraph_examples/react_agent.py` |
| **Reflexion Agent** | Self-improving agent with iterative refinement | `langgraph_examples/reflection_agent/` |
| **Deep Research** | Multi-agent system for comprehensive research | `langgraph_examples/deep_research_agent/` |
| **RAG Pipeline** | Retrieval Augmented Generation with Pinecone | `rag/` |
| **LCEL Chains** | Composable pipelines with LangChain Expression Language | `chains/` |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/aimanyounises1/LangChainTutorial.git
cd LangChainTutorial

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys (Tavily, Pinecone, etc.)

# 5. Run your first agent!
python -m langgraph_examples.react_agent
```

<details>
<summary><strong>Prerequisites</strong></summary>

- **Python 3.12+**
- **[Ollama](https://ollama.ai/)** - Local LLM inference
  - `ollama pull qwen3:30b-a3b` (or your preferred model)
- **API Keys:**
  - [Tavily](https://tavily.com/) - Web search (required)
  - [Pinecone](https://pinecone.io/) - Vector store (for RAG)
  - [LangSmith](https://smith.langchain.com/) - Tracing (optional)

</details>

---

## Features

### 1. ReAct Agent

The classic Reasoning + Acting pattern for tool-augmented LLMs.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=[search_tool])
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather?"}]})
```

```
Think → Act → Observe → Repeat
```

---

### 2. Reflexion Agent

Self-improving agent that critiques and refines its own responses.

```
┌─────────┐     ┌───────────────┐     ┌─────────┐
│  Draft  │────▶│ Execute Tools │────▶│ Reviser │──┐
└─────────┘     └───────────────┘     └─────────┘  │
     ▲                                             │
     └─────────────────────────────────────────────┘
                   (loop until quality threshold)
```

**Key Features:**
- Structured output with Pydantic models
- SQLite checkpointing for state persistence
- Built-in retry policies for resilience

```bash
python -m langgraph_examples.reflection_agent.main
```

---

### 3. Deep Research Agent

Multi-agent system for comprehensive, citation-backed research.

```
                    ┌──────────────┐
                    │   Planner    │  ← Creates research plan
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
              ┌────▶│  Researcher  │  ← Executes searches
              │     └──────┬───────┘
              │            │
              │            ▼
              │     ┌──────────────┐
              │     │ Synthesizer  │  ← Integrates findings
              │     └──────┬───────┘
              │            │
              │            ▼
              │     ┌──────────────┐
              │     │    Critic    │  ← Evaluates quality
              │     └──────┬───────┘
              │            │
              │     ┌──────┴───────┐
              │     │  Continue?   │
              │     └──────┬───────┘
              │       YES  │  NO
              └────────────┘   │
                               ▼
                    ┌──────────────┐
                    │    Report    │  ← Generates final output
                    └──────────────┘
```

**Quality Metrics:**
- Coverage Score - Sub-questions addressed
- Depth Score - Analysis quality
- Citation Density - Sources per section
- Completeness - Overall quality

```bash
# Standalone
python -m langgraph_examples.deep_research_agent.main "Your research query"

# With LangGraph Studio
langgraph dev --config langgraph_examples/deep_research_agent/langgraph.json
```

---

### 4. RAG Pipeline

Retrieval Augmented Generation with multiple ingestion strategies.

| Method | Description | Use Case |
|--------|-------------|----------|
| **Basic** | Load → Split → Embed → Store | Local documents |
| **Tavily Crawl** | Deep website scraping | Web content |
| **Tavily Extract** | Targeted content extraction | Specific pages |

```bash
# Ingest documents
python -m rag.ingestion

# Query with RAG
python -m rag.retrieval
```

---

## Project Structure

<details>
<summary><strong>Click to expand</strong></summary>

```
LangChainTutorial/
│
├── langgraph_examples/           # Modern LangGraph agents
│   ├── react_agent.py            # Prebuilt ReAct agent
│   ├── custom_graph.py           # Custom graph workflows
│   ├── reflection.py             # Basic reflection loop
│   │
│   ├── reflection_agent/         # Advanced Reflexion
│   │   ├── main.py               # Graph orchestration
│   │   ├── schemas.py            # Pydantic models
│   │   ├── chains.py             # LLM chains
│   │   └── tools_executor.py     # Tool execution
│   │
│   └── deep_research_agent/      # Multi-agent research
│       ├── graph.py              # StateGraph orchestration
│       ├── schemas.py            # Research models
│       ├── main.py               # CLI entry point
│       └── agents/               # Specialized agents
│           ├── planner.py
│           ├── researcher.py
│           ├── synthesizer.py
│           ├── critic.py
│           └── report_generator.py
│
├── agents/                       # Classic LangChain agents
│   ├── react_agent.py
│   └── search_agent.py
│
├── rag/                          # RAG implementation
│   ├── ingestion.py
│   ├── ingestion_with_tavily_rag.py
│   └── retrieval.py
│
├── chains/                       # LCEL examples
│   └── lcel_structured_example.py
│
├── core/                         # Shared utilities
│   ├── schemas.py
│   └── tools.py
│
└── docs/                         # Documentation
    ├── LCEL_EXPLANATION.md
    └── QUICK_REFERENCE.md
```

</details>

---

## Environment Variables

Create a `.env` file with:

```env
# Required
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=langchain-rag

# Optional (for tracing)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangChain Tutorial
```

---

## Examples

<details>
<summary><strong>Basic ReAct Agent</strong></summary>

```python
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from core.tools import search_tool

llm = ChatOllama(model="qwen3:30b-a3b")
agent = create_react_agent(model=llm, tools=[search_tool])

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is LangGraph?"}]
})
print(result["messages"][-1].content)
```

</details>

<details>
<summary><strong>Structured Output</strong></summary>

```python
from pydantic import BaseModel
from typing import List

class ResearchResult(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

llm_structured = llm.with_structured_output(ResearchResult)
result = llm_structured.invoke("Explain quantum computing")
print(result.answer)
print(result.sources)
```

</details>

<details>
<summary><strong>Custom Graph</strong></summary>

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    iteration: int

def process_node(state: State):
    return {"iteration": state["iteration"] + 1}

graph = StateGraph(State)
graph.add_node("process", process_node)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"messages": [], "iteration": 0})
```

</details>

---

## LangChain vs LangGraph

| Aspect | LangChain (Classic) | LangGraph (Modern) |
|--------|--------------------|--------------------|
| **Architecture** | Sequential chains | Graph-based workflows |
| **State Management** | Limited | Built-in state machine |
| **Control Flow** | Linear | Conditional, loops, branches |
| **Persistence** | Manual | Built-in checkpointing |
| **Best For** | Simple pipelines | Complex agent workflows |

**Recommendation:** Use LangGraph for new projects. It provides better control flow, state management, and debugging capabilities.

---

## Documentation

- [LCEL Explanation](docs/LCEL_EXPLANATION.md) - Deep dive into LangChain Expression Language
- [Quick Reference](docs/QUICK_REFERENCE.md) - Common patterns and snippets

## External Resources

- [LangChain Documentation](https://python.langchain.com/docs/tutorials/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Tavily API](https://docs.tavily.com/)

---

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built by <a href="https://github.com/aimanyounises1">Aiman Younises</a></strong>
</p>

<p align="center">
  <a href="https://github.com/aimanyounises1/LangChainTutorial/stargazers">
    <img src="https://img.shields.io/github/stars/aimanyounises1/LangChainTutorial?style=social" alt="Stars"/>
  </a>
  <a href="https://github.com/aimanyounises1/LangChainTutorial/network/members">
    <img src="https://img.shields.io/github/forks/aimanyounises1/LangChainTutorial?style=social" alt="Forks"/>
  </a>
</p>

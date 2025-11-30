# LangChain & LangGraph Tutorial

A comprehensive tutorial project demonstrating modern LangChain and LangGraph patterns including agents, chains, RAG, reflection agents, and graph-based workflows.

## Project Structure

```
LangChainTutorial/
├── agents/                              # Classic LangChain agents
│   ├── react_agent.py                   # ReAct agent with AgentExecutor
│   ├── react_agent_2.py                 # Alternative ReAct implementation
│   ├── react_search_agent.py            # ReAct with LCEL output parsing
│   └── search_agent.py                  # Search agent with structured output
│
├── langgraph_examples/                  # LangGraph agents (modern approach)
│   ├── react_agent.py                   # Prebuilt ReAct agent
│   ├── custom_graph.py                  # Custom graph-based workflow
│   ├── structured_output.py             # Agent with typed responses
│   ├── main.py                          # ReAct with function calling
│   ├── nodes.py                         # Node definitions for graphs
│   ├── react.py                         # LLM and tools configuration
│   ├── chain.py                         # Prompt chains for reflection
│   ├── reflection.py                    # Basic reflection graph (generate → reflect)
│   │
│   └── reflection_agent/                # Advanced Reflexion Agent
│       ├── main.py                      # Graph: draft → execute_tools → revise
│       ├── schemas.py                   # AnswerQuestion, ReviseAnswer, Reflection
│       ├── chains.py                    # Actor/Reviser prompt chains
│       └── tools_executor.py            # Tavily search tool execution
│
├── chains/                              # LCEL chain examples
│   └── lcel_structured_example.py       # Modern LCEL patterns
│
├── rag/                                 # RAG implementation
│   ├── ingestion.py                     # Document loading & indexing
│   ├── ingestion_with_tavily_rag.py     # Advanced: Tavily Crawl/Map/Extract
│   └── retrieval.py                     # Query & retrieval chains
│
├── core/                                # Shared utilities
│   ├── schemas.py                       # Pydantic models & prompts
│   └── tools.py                         # LangChain tools
│
├── docs/                                # Documentation
│   ├── LCEL_EXPLANATION.md              # LCEL deep dive
│   └── QUICK_REFERENCE.md               # Quick reference guide
│
├── main.py                              # Basic LLM example
├── pyproject.toml                       # Dependencies
└── .env.example                         # Environment template
```

## LangChain vs LangGraph

| Feature | LangChain (Classic) | LangGraph (Modern) |
|---------|--------------------|--------------------|
| **Architecture** | Sequential chains | Graph-based workflows |
| **State** | Limited | Built-in state management |
| **Control Flow** | Linear | Conditional branching, loops |
| **Best For** | Simple pipelines | Complex agent workflows |
| **Location** | `agents/` | `langgraph_examples/` |

## Features

### 1. LangGraph Agents (`langgraph_examples/`)

Modern graph-based agents using LangGraph 0.3+:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search_web],
    response_format=ResponseSchema,  # Optional structured output
)
result = agent.invoke({"messages": [{"role": "user", "content": "query"}]})
```

**Files:**
- `react_agent.py` - Prebuilt ReAct agent with Tavily search
- `custom_graph.py` - Custom graph with nodes, edges, and state
- `structured_output.py` - Typed responses with Pydantic
- `main.py` - ReAct pattern with function calling
- `reflection.py` - Basic reflection loop (generate → reflect → repeat)

### 2. Reflexion Agent (`langgraph_examples/reflection_agent/`)

Advanced self-improving agent that uses reflection to iteratively enhance responses:

```
┌─────────┐     ┌───────────────┐     ┌─────────┐
│  draft  │────▶│ execute_tools │────▶│ reviser │
└─────────┘     └───────────────┘     └────┬────┘
                       ▲                    │
                       │                    │
                       └────────────────────┘
                         (loop until done)
```

**How it works:**
1. **Draft** - First responder generates initial answer with reflection
2. **Execute Tools** - Runs Tavily search queries to gather information
3. **Revise** - Improves answer based on search results and self-critique
4. **Loop** - Repeats until max iterations reached

**Files:**
- `main.py` - Graph orchestration with MessageGraph
- `schemas.py` - `AnswerQuestion`, `ReviseAnswer`, `Reflection` models
- `chains.py` - Actor and Reviser prompt templates
- `tools_executor.py` - Tavily search execution

### 3. Classic LangChain Agents (`agents/`)

Traditional AgentExecutor-based agents:

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "query"})
```

### 4. LCEL Chains (`chains/`)

LangChain Expression Language for composable pipelines:

```python
# Build chain with pipe operator
chain = prompt | llm.with_structured_output(Schema) | parser

# Execute
result = chain.invoke({"query": "input"})
```

### 5. RAG Pipeline (`rag/`)

Retrieval Augmented Generation with Pinecone:

- **Basic Ingestion** (`ingestion.py`): Load → Split → Embed → Store
- **Advanced Ingestion** (`ingestion_with_tavily_rag.py`):
  - Tavily Crawl for deep website scraping
  - Tavily Map for sitemap discovery
  - Tavily Extract for content extraction
  - Async batch processing for performance
- **Retrieval** (`retrieval.py`): Query → Retrieve → Generate

### 6. Core Utilities (`core/`)

Shared components:
- `schemas.py` - `AgentResponse`, `Source`, `REACT_PROMPT_TEMPLATE`
- `tools.py` - `search_tool` for Tavily

## Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.ai/)** with models:
  - `qwen3:30b-a3b` (or preferred model)
  - `qwen3-embedding:latest` (for RAG)
- **API Keys:**
  - [Tavily](https://tavily.com/) - Web search
  - [Pinecone](https://pinecone.io/) - Vector store
  - [LangSmith](https://smith.langchain.com/) - Tracing (optional)

## Installation

```bash
# Clone
git clone https://github.com/aimanyounises1/LangChainTutorial.git
cd LangChainTutorial

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install
pip install -e .
# Or with uv:
uv sync

# Environment
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

```env
# Required
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=langchain-rag

# Optional (tracing)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=LangChain Tutorial
```

## Usage

### LangGraph Agents (Recommended)

```bash
# Prebuilt ReAct agent
python -m langgraph_examples.react_agent

# Custom graph workflow
python -m langgraph_examples.custom_graph

# ReAct with function calling
python -m langgraph_examples.main

# Basic reflection loop
python -m langgraph_examples.reflection

# Advanced Reflexion agent
python -m langgraph_examples.reflection_agent.main
```

### Classic LangChain Agents

```bash
python -m agents.react_agent
python -m agents.search_agent
```

### LCEL Chains

```bash
python -m chains.lcel_structured_example
```

### RAG Pipeline

```bash
# Basic ingestion
python -m rag.ingestion

# Advanced ingestion with Tavily
python -m rag.ingestion_with_tavily_rag

# Query
python -m rag.retrieval
```

## Key Concepts

### ReAct Pattern

Reasoning and Acting loop:
1. **Thought** - Analyze the problem
2. **Action** - Choose and execute a tool
3. **Observation** - Process tool result
4. **Repeat** until final answer

### Reflexion Pattern

Self-improving agent loop:
1. **Draft** - Generate initial response with self-critique
2. **Search** - Research to address identified gaps
3. **Revise** - Improve answer with citations
4. **Repeat** until quality threshold met

### LangGraph State Machine

```
START → agent → [tools → agent]* → END
         ↓           ↑
    (decide)    (loop back)
```

### LCEL Pipe Operator

```python
# Build BEFORE execution
chain = step1 | step2 | step3

# Then execute
result = chain.invoke(input)
```

### Structured Outputs

```python
class Response(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

llm.with_structured_output(Response)
```

## Dependencies

```toml
langchain >= 1.1.0
langchain-community >= 0.4.1
langchainhub
langchain-ollama >= 1.0.0
langchain-openai >= 1.1.0
langchain-pinecone >= 0.2.13
langchain-tavily >= 0.2.13
langchain-text-splitters >= 1.0.0
langgraph >= 0.3.0
grandalf >= 0.8
```

## Documentation

- [LCEL Explanation](docs/LCEL_EXPLANATION.md) - Understanding LCEL
- [Quick Reference](docs/QUICK_REFERENCE.md) - Common patterns

## Resources

- [LangChain Docs](https://python.langchain.com/docs/tutorials/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangChain Architecture](https://python.langchain.com/docs/concepts/architecture/)

## License

MIT

## Author

Aiman Younises

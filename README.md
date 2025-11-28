# LangChain & LangGraph Tutorial

A comprehensive tutorial project demonstrating modern LangChain and LangGraph patterns including agents, chains, RAG, and graph-based workflows.

## Project Structure

```
LangChainTutorial/
├── agents/                        # Classic LangChain agents
│   ├── react_agent.py             # ReAct agent with AgentExecutor
│   ├── react_agent_2.py           # Alternative ReAct implementation
│   ├── react_search_agent.py      # ReAct with LCEL output parsing
│   └── search_agent.py            # Search agent with structured output
│
├── langgraph/                     # LangGraph agents (modern approach)
│   ├── react_agent.py             # Prebuilt ReAct agent
│   ├── custom_graph.py            # Custom graph-based workflow
│   └── structured_output.py       # Agent with typed responses
│
├── chains/                        # LCEL chain examples
│   └── lcel_structured_example.py # Modern LCEL patterns
│
├── rag/                           # RAG implementation
│   ├── ingestion.py               # Document loading & indexing
│   └── retrieval.py               # Query & retrieval chains
│
├── core/                          # Shared utilities
│   ├── schemas.py                 # Pydantic models & prompts
│   └── tools.py                   # LangChain tools
│
├── docs/                          # Documentation
│   ├── LCEL_EXPLANATION.md        # LCEL deep dive
│   └── QUICK_REFERENCE.md         # Quick reference guide
│
├── main.py                        # Basic LLM example
├── pyproject.toml                 # Dependencies
└── .env.example                   # Environment template
```

## LangChain vs LangGraph

| Feature | LangChain (Classic) | LangGraph (Modern) |
|---------|--------------------|--------------------|
| **Architecture** | Sequential chains | Graph-based workflows |
| **State** | Limited | Built-in state management |
| **Control Flow** | Linear | Conditional branching, loops |
| **Best For** | Simple pipelines | Complex agent workflows |
| **Location** | `agents/` | `langgraph/` |

## Features

### 1. LangGraph Agents (`langgraph/`)

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

### 2. Classic LangChain Agents (`agents/`)

Traditional AgentExecutor-based agents:

```python
from langchain_classic.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "query"})
```

### 3. LCEL Chains (`chains/`)

LangChain Expression Language for composable pipelines:

```python
# Build chain with pipe operator
chain = prompt | llm.with_structured_output(Schema) | parser

# Execute
result = chain.invoke({"query": "input"})
```

### 4. RAG Pipeline (`rag/`)

Retrieval Augmented Generation with Pinecone:

- **Ingestion**: Load → Split → Embed → Store
- **Retrieval**: Query → Retrieve → Generate

### 5. Core Utilities (`core/`)

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
python -m langgraph.react_agent

# Custom graph workflow
python -m langgraph.custom_graph

# Structured output agent
python -m langgraph.structured_output
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
# Ingest documents
python -m rag.ingestion

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
langchain-ollama >= 1.0.0
langchain-pinecone >= 0.2.13
langchain-tavily >= 0.2.13
langgraph >= 0.3.0
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
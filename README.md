# LangChain Tutorial

A comprehensive tutorial project demonstrating LangChain patterns including agents, chains, and RAG (Retrieval Augmented Generation).

## Project Structure

```
LangChainTutorial/
├── agents/                        # Agent implementations
│   ├── react_agent.py             # ReAct agent with Tavily search
│   ├── react_agent_2.py           # Alternative ReAct implementation
│   ├── react_search_agent.py      # ReAct with LCEL output parsing
│   └── search_agent.py            # Simple search agent with structured output
├── chains/                        # LCEL chain examples
│   └── lcel_structured_example.py # Modern LCEL patterns with structured output
├── core/                          # Shared utilities
│   ├── schemas.py                 # Pydantic models and prompt templates
│   └── tools.py                   # LangChain tools (Tavily search)
├── rag/                           # RAG implementation
│   ├── ingestion.py               # Document loading and Pinecone indexing
│   └── retrieval.py               # Query and retrieval chains
├── docs/                          # Documentation
│   ├── LCEL_EXPLANATION.md        # Deep dive into LCEL concepts
│   └── QUICK_REFERENCE.md         # Quick lookup guide
├── main.py                        # Basic LLM example with Ollama
├── pyproject.toml                 # Project dependencies
└── .env.example                   # Environment variables template
```

## Features

### 1. Agents (agents/)

Different agent patterns using LangChain:

- ReAct Pattern: Reasoning and acting agents that think step-by-step
  - Search Agents: Web search using the Tavily API
  - Structured Outputs: Pydantic models for type-safe responses

Example: Running a ReAct agent

```python
from agents.react_agent import main
main()
```

### 2. LCEL Chains (chains/)

Modern LangChain Expression Language patterns:

- Build composable chains with the pipe operator `|`
- Use `with_structured_output()` for guaranteed schema compliance
- Support for parallel and sequential chain execution

Example: LCEL chain with structured output

```python
chain = prompt | llm.with_structured_output(ResponseSchema)
result = chain.invoke({"query": "What is Bitcoin?"})
```

### 3. RAG (rag/)

Retrieval Augmented Generation with Pinecone:

- Ingestion: Load documents, split into chunks, embed with Ollama, store in Pinecone
- Retrieval: Query the vector store and generate answers with context

### 4. Core Utilities (core/)

Shared components:

- Schemas: `AgentResponse`, `Source` Pydantic models
- Tools: `search_tool` for Tavily web search
- Prompts: `REACT_PROMPT_TEMPLATE` for ReAct agents

## Prerequisites

- Python 3.12+
- Ollama (https://ollama.ai/) with models installed:
  - `qwen3:30b-a3b` (or your preferred model)
  - `qwen3-embedding:latest` (for RAG embeddings)
- API Keys:
  - Tavily API key (for web search)
  - Pinecone API key (for RAG)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aimanyounises1/LangChainTutorial.git
cd LangChainTutorial
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
# Or with uv:
uv sync
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

Create a `.env` file with:

```env
# Tavily API (web search)
TAVILY_API_KEY=your_tavily_api_key

# Pinecone (vector store for RAG)
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_index_name

# Optional: LangSmith (tracing)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
```

## Usage

### Basic LLM Example

```bash
python main.py
```

### Run a ReAct Agent

```bash
python -m agents.react_agent
```

### Run Search Agent

```bash
python -m agents.search_agent
```

### LCEL Structured Output Demo

```bash
python -m chains.lcel_structured_example
```

### RAG Pipeline

```bash
# First, ingest documents
python -m rag.ingestion

# Then query
python -m rag.retrieval
```

## Key Concepts

### LCEL (LangChain Expression Language)

Build chains declaratively using the pipe operator:

```python
# Build chain BEFORE execution
chain = prompt | llm | output_parser

# Execute the chain
result = chain.invoke({"input": "query"})
```

Important: The pipe operator `|` works with Runnables, not executed data. See `docs/LCEL_EXPLANATION.md` for details.

### ReAct Pattern

Agents that reason step-by-step:

1. Thought: Analyze the question
2. Action: Choose a tool to use
3. Observation: See the tool result
4. Repeat until final answer

### Structured Outputs

Get type-safe responses using Pydantic:

```python
class AgentResponse(BaseModel):
    answer: str
    sources: List[Source]

llm_structured = llm.with_structured_output(AgentResponse)
```

## Dependencies

- `langchain` - Core LangChain framework
- `langchain-ollama` - Ollama integration
- `langchain-community` - Community integrations
- `langchain-pinecone` - Pinecone vector store
- `langchain-tavily` - Tavily search integration
- `langchain-text-splitters` - Document chunking
- `langchain-openai` - OpenAI integration (optional)

## Documentation

- [LCEL Explanation](docs/LCEL_EXPLANATION.md) - Understanding LangChain Expression Language
- [Quick Reference](docs/QUICK_REFERENCE.md) - Common patterns and fixes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Author

Aiman Younises

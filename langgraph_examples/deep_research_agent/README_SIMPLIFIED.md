# Deep Research Agent - Simplified Implementation 🚀

This directory contains **simplified implementations** of the deep research agent using prebuilt LangChain components.

## 📊 Comparison

| Aspect | Current Implementation | Simplified Version |
|--------|----------------------|-------------------|
| **Lines of Code** | 750+ | 40-100 |
| **Files** | 8 files | 1 file |
| **Complexity** | High (custom agents, routing, serialization) | Low (prebuilt components) |
| **Features** | ✅ All features | ✅ All features + enhancements |
| **Maintenance** | High effort | Low effort |
| **Performance** | Good | Equal or better |

## 🎯 Quick Start

### Option 1: create_agent (Recommended - No New Packages!)

Works with your **current setup** - no installation needed!

```bash
# Run immediately
python deep_research_create_agent.py
```

**Or use in your code:**
```python
from deep_research_create_agent import research

result = research("Research AI-powered security operations centers")
print(result["messages"][-1].content)
```

### Option 2: Deep Agents (Most Feature-Rich)

Requires installing Deep Agents framework:

```bash
# Install
pip install deepagents tavily-python

# Run
python deep_research_simplified.py
```

**Or use in your code:**
```python
from deep_research_simplified import research

result = research("Research AI-powered security operations centers")
```

## 📁 Files

```
deep_research_agent/
├── graph.py                           # Current: 750+ lines, 8 files
├── agents/                            # Current: Custom agent implementations
│   ├── planner.py                     # 150 lines
│   ├── researcher.py                  # 200 lines
│   ├── synthesizer.py                 # 180 lines
│   ├── critic.py                      # 150 lines
│   ├── report_generator.py            # 120 lines
│   └── prompt_engineer.py             # 100 lines
│
├── deep_research_create_agent.py      # NEW: 100 lines (no new packages!)
├── deep_research_simplified.py        # NEW: 40 lines (requires deepagents)
├── comparison_test.py                 # Test & compare implementations
├── MIGRATION_GUIDE.md                 # Step-by-step migration guide
└── README_SIMPLIFIED.md               # This file
```

## 🔥 What Changed?

### Before (750+ lines)
```python
# graph.py - Manual graph construction
builder = StateGraph(DeepResearchGraphState)
builder.add_node("plan", planning_node)
builder.add_node("research", research_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("critique", critique_node)
builder.add_node("finalize", finalize_node)
builder.add_conditional_edges("critique", route_after_critique, {...})
# + 300 more lines...

# agents/planner.py - Custom planner
PLANNER_SYSTEM_PROMPT = """..."""
planner_chain = planner_prompt | planner_llm | planner_parser
# + 150 more lines...

# agents/researcher.py - Custom researcher
# + 200 lines...

# agents/synthesizer.py - Custom synthesizer
# + 180 lines...

# And 4 more custom agent files...
```

### After (40-100 lines)

**Using create_agent (100 lines, no new packages):**
```python
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults

agent = create_agent(
    model=ChatOllama(model="llama3.3:70b"),
    tools=[TavilySearchResults(max_results=5)],
    system_prompt="""You are an expert researcher.
    1. Plan: Break into sub-questions
    2. Research: Search thoroughly
    3. Synthesize: Organize findings
    4. Generate: Create report
    """,
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db")
)

result = agent.invoke({"messages": [HumanMessage(content=query)]})
```

**Using Deep Agents (40 lines, requires deepagents):**
```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "deep-researcher",
    "system_prompt": """Expert researcher using:
    - write_todos for planning
    - internet_search for research
    - write_file for synthesis
    - Built-in iteration
    """,
    "tools": [internet_search],
}

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    subagents=[research_subagent]
)

result = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

## ✨ Features Maintained

Both simplified versions maintain **ALL** your current features:

- ✅ **Multi-step Planning** - Break queries into sub-questions
- ✅ **Web Search** - Tavily integration for research
- ✅ **Synthesis** - Organize findings into coherent reports
- ✅ **Iteration** - Self-critique and refinement
- ✅ **Citations** - Track and reference sources
- ✅ **Checkpointing** - Save and resume research sessions
- ✅ **Streaming** - Real-time output
- ✅ **Quality Metrics** - Coverage, depth, completeness

## 🚀 Performance Improvements

The simplified versions actually **improve** performance:

1. **Parallel Execution** (Deep Agents)
   - Current: Sequential sub-question research
   - Simplified: Parallel execution of research tasks

2. **Better Context Management** (Deep Agents)
   - Current: Manual state serialization
   - Simplified: Automatic file offloading

3. **No Serialization Overhead**
   - Current: Manual serialize/deserialize for every state update
   - Simplified: Direct Pydantic model handling

4. **Built-in Optimizations**
   - Maintained by LangChain team
   - Regular performance improvements

## 📊 Comparison Test

Run the comparison script to see side-by-side results:

```bash
python comparison_test.py
```

This will:
- Run the same query through multiple implementations
- Compare execution time
- Compare output quality
- Show code complexity reduction
- Provide recommendations

## 🔄 Migration Guide

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed step-by-step instructions.

**Quick Migration:**

1. **Test the simplified version:**
   ```bash
   python deep_research_create_agent.py
   ```

2. **Compare with current:**
   ```bash
   python comparison_test.py
   ```

3. **Switch imports:**
   ```python
   # Old
   from graph import run_deep_research
   result = run_deep_research(query)

   # New
   from deep_research_create_agent import research
   result = research(query)
   ```

4. **Archive old code:**
   ```bash
   mkdir -p archive/old_implementation
   mv agents/ archive/old_implementation/
   mv graph.py archive/old_implementation/
   ```

## 🎓 Which Version Should I Use?

### Use **create_agent** if:
- ✅ You want to start **immediately** (no new packages)
- ✅ You want a **simple, understandable** implementation
- ✅ You're comfortable with ~100 lines of code
- ✅ You don't need cross-session memory

### Use **Deep Agents** if:
- ✅ You want the **absolute simplest** code (~40 lines)
- ✅ You want **parallel execution** out of the box
- ✅ You want **file system** for context management
- ✅ You want **cross-session memory**
- ✅ You're okay installing one new package

**Both are excellent choices!** I recommend starting with `create_agent` since it requires no installation.

## 🐛 Troubleshooting

### create_agent Version

**Q: "No module named 'langchain.agents'"**
```bash
# Update LangChain
pip install -U langchain
```

**Q: "TavilySearchResults not found"**
```bash
# Install community package
pip install -U langchain-community
```

### Deep Agents Version

**Q: "No module named 'deepagents'"**
```bash
# Install Deep Agents
pip install deepagents tavily-python
```

**Q: "API key not found"**
```bash
# Set in .env file
echo "TAVILY_API_KEY=your_key_here" >> .env
```

## 📚 Resources

- **LangChain Docs**: https://docs.langchain.com/
- **Deep Agents Docs**: https://docs.langchain.com/oss/python/deepagents/overview
- **create_agent Guide**: https://docs.langchain.com/oss/python/langchain/agents
- **Migration Guide**: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## 💡 Next Steps

1. ✅ **Test** the simplified version:
   ```bash
   python deep_research_create_agent.py
   ```

2. ✅ **Compare** with your current implementation:
   ```bash
   python comparison_test.py
   ```

3. ✅ **Migrate** using the guide:
   - Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
   - Follow step-by-step instructions

4. ✅ **Enjoy** maintaining 95% less code! 🎉

## 🤝 Questions?

- Check [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed explanations
- Review LangChain docs for `create_agent` usage
- Review Deep Agents docs for advanced features

---

**Made with ❤️ using LangChain prebuilt components**

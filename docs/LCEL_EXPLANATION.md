# Understanding LCEL (LangChain Expression Language) and Structured Outputs

## What is LCEL?

**LCEL (LangChain Expression Language)** is a declarative way to compose LangChain components using the pipe operator `|`. It allows you to build processing pipelines in a clean, readable way.

### Key Concept: Runnables vs Results

```python
# ✅ CORRECT: Chain of Runnables (not executed yet)
chain = prompt | llm | output_parser
result = chain.invoke(input)  # Execute here

# ❌ WRONG: Trying to pipe already-executed data
result = llm.invoke(input)  # Already executed → returns data
chain = result | parser  # ERROR! Can't pipe data
```

## The Pipe Operator `|`

The pipe operator chains **Runnables** (things that CAN be executed), not **data** (things that HAVE BEEN executed).

### Anatomy of LCEL Chain:

```
Component 1    Component 2    Component 3
(Runnable)     (Runnable)     (Runnable)
    |              |              |
    v              v              v
[Prompt]  ---->  [LLM]  ---->  [Parser]  ----> INVOKE ----> Result
```

### What You Were Trying (Incorrect):

```python
result = agent_executor.invoke(...)  # ← This EXECUTES the agent
chain = result | extract_output | structured_output  # ❌ result is DATA, not a Runnable
```

**Why it failed:**
- `result` is a **dictionary** (already executed data)
- The pipe operator `|` only works with **Runnables**
- You're trying to pipe water that's already flowed through the pipe!

---

## Two Approaches to Structured Outputs

### Approach 1: Post-Processing (What We Implemented)

**When to use:** When working with AgentExecutor from langchain_classic

```python
# Step 1: Execute the agent
result = agent_executor.invoke({"input": "query"})

# Step 2: Process the result (post-processing, NOT LCEL)
formatted_response = extract_sources_from_result(result)
```

**Pros:**
- Simple and straightforward
- Works with classic AgentExecutor
- Easy to debug

**Cons:**
- Requires manual processing
- Sources extracted after execution

---

### Approach 2: Using `with_structured_output()` (Modern LangChain)

**When to use:** When building custom chains with direct LLM access

```python
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    url: str

class AgentResponse(BaseModel):
    answer: str
    sources: List[Source]

# Build chain with structured output BEFORE execution
llm = ChatOllama(model="qwen3:30b-a3b")
structured_llm = llm.with_structured_output(AgentResponse)

# Now build the LCEL chain
chain = prompt | structured_llm  # ← This is a Runnable chain

# Execute the chain
result = chain.invoke({"input": "What is Bitcoin price?"})
# result is now an AgentResponse object!
```

**Pros:**
- LLM automatically returns structured data
- Type-safe responses
- Modern LangChain pattern

**Cons:**
- Doesn't work directly with AgentExecutor
- Requires schema definition upfront

---

## Understanding the Error You Got

```
TypeError: Expected a Runnable, callable or dict.
Instead got an unsupported type: <class 'str'>
```

**What happened:**

1. You invoked the agent: `result = agent_executor.invoke(...)`
2. `result` became a **dict** (data)
3. You tried: `chain = result | extract_output`
4. LangChain tried to convert `result` to a Runnable
5. Failed because **you can't convert executed data into a Runnable**

---

## When to Use Each Pattern

### Use Post-Processing When:
- Working with `AgentExecutor` from langchain_classic
- Following older LangChain tutorials (like your Udemy course)
- You need to extract information from intermediate steps
- You want simple, straightforward code

### Use LCEL Chains When:
- Building custom prompts → LLM → parser pipelines
- Creating reusable components
- You want composable, declarative code
- Working with modern LangChain patterns

### Use `with_structured_output()` When:
- You want guaranteed structured responses
- Building production applications with type safety
- Using modern LLM features
- You need the LLM to return specific formats (JSON, Pydantic models)

---

## Common LCEL Patterns

### Pattern 1: Simple Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOllama(model="qwen3:30b-a3b")
parser = StrOutputParser()

# Build the chain (not executed yet)
chain = prompt | llm | parser

# Execute it
result = chain.invoke({"topic": "programming"})
```

### Pattern 2: Branching with RunnableParallel
```python
from langchain_core.runnables import RunnableParallel

# Build parallel branches
chain = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain
)

result = chain.invoke({"topic": "AI"})
# result = {"joke": "...", "poem": "..."}
```

### Pattern 3: Conditional Routing
```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x["input"], code_chain),
    (lambda x: "joke" in x["input"], joke_chain),
    default_chain
)

result = branch.invoke({"input": "tell me a joke"})
```

---

## Key Takeaways

1. **LCEL chains must be built BEFORE execution**
2. **The pipe operator `|` only works with Runnables**
3. **You cannot pipe already-executed data**
4. **For AgentExecutor, use post-processing instead of LCEL**
5. **For custom chains, LCEL is powerful and clean**

---

## Your Fixed Code Explained

```python
def main():
    # Create the agent executor (this is a Runnable)
    agent_executor = create_react_agent_executor()

    # Execute it (now result is DATA, not a Runnable)
    result = agent_executor.invoke({
        "input": "What is the current price of Bitcoin?"
    })

    # Process the data (post-processing, NOT LCEL)
    formatted_response = extract_sources_from_result(result)

    # Display results
    print(formatted_response.answer)
    for source in formatted_response.source:
        print(source.url)
```

**Why this works:**
- We execute first, THEN process
- We don't try to pipe data through chains
- We use simple function calls for post-processing

---

## Additional Resources

- [Official LCEL Documentation](https://python.langchain.com/docs/concepts/lcel/)
- [Structured Outputs Guide](https://python.langchain.com/docs/concepts/structured_outputs/)
- [LangGraph ReAct with Structured Output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)

---

## Summary

**The Error:** You tried to pipe executed data (`result` dictionary) through an LCEL chain.

**The Fix:** Process the result using regular Python functions after execution.

**The Lesson:** LCEL chains are blueprints (built before execution), not processing steps for already-executed data.

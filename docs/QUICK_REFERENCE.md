# Quick Reference: LCEL and Structured Outputs

## Your Error - Quick Explanation

**What you tried:**
```python
result = agent_executor.invoke(...)  # ← Already executed (returns dict)
chain = result | extract | format    # ❌ ERROR: Can't pipe a dict!
```

**Why it failed:**
- The pipe operator `|` only works with **Runnables** (things that CAN be executed)
- `result` is **data** (already executed), not a Runnable
- You're trying to pipe water that already flowed through the pipe

**The fix:**
```python
# Execute first
result = agent_executor.invoke(...)

# Then process (not piping, just calling a function)
formatted = extract_sources_from_result(result)
```

---

## When to Use Each Pattern

### Pattern 1: AgentExecutor (react_agent.py) ✅ FIXED
**Use when:** Following Udemy course with classic LangChain patterns

```python
# 1. Execute
result = agent_executor.invoke({"input": "query"})

# 2. Post-process (NOT an LCEL chain!)
formatted = extract_sources_from_result(result)
```

### Pattern 2: LCEL with Structured Output (lcel_structured_example.py)
**Use when:** Building modern LangChain applications

```python
# 1. Build chain FIRST (not executed yet)
chain = prompt | llm.with_structured_output(Schema)

# 2. Then execute
result = chain.invoke(input)  # result is Schema object automatically
```

---

## Golden Rules

1. **Build chains BEFORE executing them**
   ```python
   ✅ chain = step1 | step2 | step3  # Build first
   ✅ result = chain.invoke(input)  # Execute second

   ❌ result = step1.invoke(input)  # Executed
   ❌ chain = result | step2        # Can't pipe data!
   ```

2. **The pipe operator only works with Runnables**
   ```python
   ✅ prompt | llm | parser          # All Runnables
   ❌ dict | parser                  # dict is not a Runnable
   ❌ string | parser                # string is not a Runnable
   ```

3. **For AgentExecutor, use post-processing**
   ```python
   ✅ result = executor.invoke(...)
   ✅ formatted = process_result(result)
   ```

---

## Files in This Project

1. **react_agent.py** - Classic ReAct pattern with AgentExecutor (FIXED)
2. **lcel_structured_example.py** - Modern LCEL examples with structured output
3. **LCEL_EXPLANATION.md** - Comprehensive guide to LCEL concepts
4. **QUICK_REFERENCE.md** - This file (quick lookup)

---

## Common Mistakes

### Mistake 1: Piping executed data
```python
❌ result = llm.invoke(input) | parser
✅ chain = llm | parser
✅ result = chain.invoke(input)
```

### Mistake 2: Using LCEL with AgentExecutor
```python
❌ agent_executor | extract_sources  # AgentExecutor doesn't work like this
✅ result = agent_executor.invoke(...)
✅ formatted = extract_sources(result)
```

### Mistake 3: Confusing Runnables with data
```python
❌ {"key": "value"} | runnable  # dict is data
❌ "string" | runnable          # string is data
✅ RunnablePassthrough() | runnable  # Runnable
✅ RunnableLambda(fn) | runnable     # Runnable
```

---

## Next Steps

1. ✅ **Understand your current code** - react_agent.py now works correctly
2. 📖 **Read LCEL_EXPLANATION.md** - Deep dive into concepts
3. 🧪 **Run lcel_structured_example.py** - See LCEL in action
4. 🎯 **Practice** - Build your own chains

---

## Need Help?

- Check the official docs: https://python.langchain.com/docs/concepts/lcel/
- Review the example files in this directory
- Remember: Build chains first, execute second!

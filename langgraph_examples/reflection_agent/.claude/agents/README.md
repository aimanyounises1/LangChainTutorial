# Claude Code Sub-Agents for LangChain/LangGraph Refactoring

This directory contains specialized Claude Code sub-agents designed for refactoring
and maintaining LangChain/LangGraph code with automated documentation lookup and testing.

## Agent Architecture

```
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │   (Coordinator) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Web Searcher   │ │   Refactorer    │ │     Tester      │
│                 │ │                 │ │                 │
│ - Documentation │ │ - Code editing  │ │ - Unit tests    │
│ - API lookup    │ │ - Imports       │ │ - Validation    │
│ - Error search  │ │ - Patterns      │ │ - Error reports │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Available Agents

### 1. `web-searcher`
**Purpose**: Search LangChain/LangGraph documentation for correct API usage

**Tools**: WebSearch, WebFetch, Read

**Use When**:
- Looking up correct import paths
- Finding migration guides for deprecated APIs
- Resolving error messages
- Finding code examples from official docs

### 2. `refactorer`
**Purpose**: Apply code modifications based on documentation findings

**Tools**: Read, Edit, Write, Glob, Grep, Bash

**Use When**:
- Updating deprecated API calls
- Fixing import statements
- Refactoring to follow best practices
- Applying fixes for errors

### 3. `tester`
**Purpose**: Validate code correctness through testing

**Tools**: Read, Write, Edit, Bash, Glob, Grep

**Use When**:
- Writing unit tests for refactored code
- Running validation checks
- Verifying no runtime errors
- Checking code quality

### 4. `orchestrator`
**Purpose**: Coordinate the entire refactoring workflow

**Tools**: Read, Glob, Grep, Task

**Use When**:
- Starting a full refactoring workflow
- Managing the search → refactor → test cycle
- Handling error resolution loops

## Workflow

1. **User Request** → "Refactor this code to follow current LangChain patterns"

2. **Orchestrator** analyzes the code and identifies components to update

3. **Web Searcher** searches official documentation for:
   - Current API patterns
   - Migration guides
   - Best practices

4. **Refactorer** applies changes:
   - Updates imports
   - Modifies API calls
   - Follows documentation patterns

5. **Tester** validates:
   - Runs syntax checks
   - Executes unit tests
   - Reports any failures

6. **Error Loop** (if needed):
   - Errors go back to Web Searcher
   - Fixes applied by Refactorer
   - Re-validated by Tester

## Usage Examples

### Direct Agent Invocation
```
Use the web-searcher agent to find the correct way to use StateGraph in LangGraph
```

### Workflow Invocation
```
Use the orchestrator agent to refactor langgraph_examples/reflection_agent/main.py
to use the latest LangGraph patterns
```

### Error Resolution
```
Use the web-searcher agent to find the solution for this error:
ImportError: cannot import name 'MessageGraph' from 'langgraph.graph'
```

## Configuration

Agents are defined in YAML frontmatter with:
- `name`: Agent identifier
- `description`: When to use this agent (PROACTIVELY triggered)
- `tools`: Comma-separated list of allowed tools
- `model`: `inherit` to use the parent model

## Dependencies

These agents work with the existing LangChain/LangGraph codebase:
- Python 3.12+
- langchain >= 1.1.0
- langgraph >= 0.3.0
- pytest (for testing)

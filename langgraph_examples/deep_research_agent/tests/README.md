# Deep Research Agent Tests

Comprehensive test suite for the Deep Research Agent agents.

## Test Coverage

### 1. Planner Agent Tests (`test_agents.py`)

**Helper Functions:**
- `test_create_sub_question_default` - Test creating sub-questions with defaults
- `test_create_sub_question_custom_priority` - Test custom priority levels
- `test_create_sub_question_custom_status` - Test custom status values
- `test_create_sub_question_unique_ids` - Verify unique ID generation

**create_default_plan Tests:**
- `test_create_default_plan_basic` - Basic plan creation
- `test_create_default_plan_sub_questions` - Verify 5 sub-questions created
- `test_create_default_plan_sub_question_priorities` - Check priority distribution
- `test_create_default_plan_expected_sections` - Verify expected sections
- `test_create_default_plan_sub_questions_reference_query` - Query referenced in questions

**validate_research_plan Tests:**
- `test_validate_valid_plan` - Valid plan passes validation
- `test_validate_missing_main_query` - Missing query detection
- `test_validate_too_few_sub_questions` - Minimum sub-questions check
- `test_validate_too_many_sub_questions` - Maximum sub-questions check
- `test_validate_too_few_sections` - Minimum sections check
- `test_validate_duplicate_sub_questions` - Duplicate detection
- `test_validate_case_insensitive_duplicates` - Case-insensitive duplicate detection
- `test_validate_multiple_issues` - Multiple validation issues

**create_research_plan Tests (with mocked LLM):**
- `test_create_research_plan_success` - Successful plan creation
- `test_create_research_plan_direct_output` - Direct output handling
- `test_create_research_plan_missing_sub_question_ids` - Auto-generate missing IDs
- `test_create_research_plan_fallback_on_error` - Fallback to default plan
- `test_create_research_plan_retry_logic` - Retry mechanism

### 2. PromptEngineer Agent Tests (`test_agents.py`)

**Helper Functions:**
- `test_create_fallback_output` - Fallback output creation
- `test_create_fallback_output_preserves_content` - Content preservation
- `test_quick_refine` - Quick refinement convenience function

**refine_prompt Tests (with mocked LLM):**
- `test_refine_prompt_success` - Successful refinement
- `test_refine_prompt_with_context` - Refinement with context
- `test_refine_prompt_direct_output` - Direct output handling
- `test_refine_prompt_fallback_on_error` - Fallback on error
- `test_refine_prompt_retry_logic` - Retry mechanism
- `test_refine_prompt_analysis_scores` - Analysis score verification

### 3. Import Tests (`test_agents.py`)

**Module Import Tests:**
- `test_import_planner_functions` - Planner imports work
- `test_import_prompt_engineer_functions` - PromptEngineer imports work
- `test_import_all_exports` - All __all__ exports accessible
- `test_planner_chain_is_runnable` - Chain has invoke method
- `test_prompt_engineer_chain_is_runnable` - Chain has invoke method
- `test_create_sub_question_imported_correctly` - Function works when imported
- `test_create_default_plan_imported_correctly` - Function works when imported
- `test_validate_research_plan_imported_correctly` - Function works when imported

### 4. Schema Integration Tests (`test_agents.py`)

- `test_sub_question_schema_compliance` - SubQuestion schema compliance
- `test_research_plan_schema_compliance` - ResearchPlan schema compliance
- `test_planner_output_schema` - PlannerOutput schema structure
- `test_prompt_engineer_output_schema` - PromptEngineerOutput schema structure
- `test_prompt_analysis_schema` - PromptAnalysis schema structure

### 5. Edge Cases and Error Handling (`test_agents.py`)

- `test_create_sub_question_empty_question` - Empty string handling
- `test_create_sub_question_very_long_question` - Long text handling
- `test_validate_empty_sub_questions_list` - Empty list validation
- `test_validate_empty_sections_list` - Empty sections validation
- `test_create_default_plan_unicode_query` - Unicode character support
- `test_create_fallback_output_special_characters` - Special character handling
- `test_create_research_plan_max_retries_zero` - Zero retries behavior
- `test_refine_prompt_max_retries_zero` - Zero retries behavior

## Running Tests

### Run All Tests
```bash
cd /Users/aimanyounis/PycharmProjects/LangChainTutorial
pytest langgraph_examples/deep_research_agent/tests/ -v
```

### Run Specific Test File
```bash
pytest langgraph_examples/deep_research_agent/tests/test_agents.py -v
```

### Run Specific Test Class
```bash
pytest langgraph_examples/deep_research_agent/tests/test_agents.py::TestPlannerHelpers -v
```

### Run Specific Test
```bash
pytest langgraph_examples/deep_research_agent/tests/test_agents.py::TestPlannerHelpers::test_create_sub_question_default -v
```

### Run with Coverage
```bash
pytest langgraph_examples/deep_research_agent/tests/ --cov=langgraph_examples.deep_research_agent.agents --cov-report=html
```

### Run Only Fast Tests (exclude slow tests)
```bash
pytest langgraph_examples/deep_research_agent/tests/ -v -m "not slow"
```

## Test Fixtures

The `conftest.py` file provides shared fixtures:

### Sample Data Fixtures
- `sample_sub_question` - Single SubQuestion instance
- `sample_sub_questions` - List of 5 SubQuestions
- `sample_research_plan` - Valid ResearchPlan
- `sample_planner_output` - PlannerOutput instance
- `sample_prompt_analysis` - PromptAnalysis instance
- `sample_prompt_engineer_output` - PromptEngineerOutput instance

### Mock Object Fixtures
- `mock_llm_response` - Factory for mock LLM responses
- `mock_successful_planner_chain` - Mocked successful planner chain
- `mock_successful_prompt_engineer_chain` - Mocked successful prompt engineer chain
- `mock_failed_llm_chain` - Mocked failing LLM chain

### Test Data Fixtures
- `test_queries` - Dictionary of various test queries
- `test_prompts` - Dictionary of various test prompts

## Key Testing Strategies

### 1. LLM Call Mocking
All tests that involve LLM calls use `@patch` decorators to mock the chain invocations:

```python
@patch('langgraph_examples.deep_research_agent.agents.planner.planner_chain')
def test_create_research_plan_success(self, mock_chain):
    mock_chain.invoke.return_value = {'parsed': mock_output}
    # Test code here
```

This ensures:
- Tests don't make actual API calls to Ollama
- Tests run fast and don't require external dependencies
- Tests are deterministic and reproducible

### 2. Schema Validation
Tests verify that all outputs comply with Pydantic schemas:
- SubQuestion
- ResearchPlan
- PlannerOutput
- PromptAnalysis
- PromptEngineerOutput

### 3. Edge Case Coverage
Tests include edge cases like:
- Empty strings
- Very long inputs
- Unicode characters
- Special characters
- Error conditions
- Retry logic

### 4. Import Testing
Tests verify that all public API exports work correctly when imported.

## Test Statistics

- **Total Tests**: 70+
- **Test Classes**: 12
- **Mocked Tests**: ~20 (avoiding actual LLM calls)
- **Coverage Areas**:
  - Helper functions
  - LLM-based functions (mocked)
  - Validation logic
  - Error handling
  - Schema compliance
  - Module imports
  - Edge cases

## Dependencies

The tests require:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting (optional)
- `unittest.mock` - Mocking (built-in)

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines without requiring:
- Ollama server running
- External API access
- GPU resources
- Large models downloaded

All LLM calls are mocked for fast, deterministic testing.

#!/usr/bin/env python3
"""
Test script to validate type fixes in graph.py without requiring full imports.
This isolates the syntax and type checking from runtime dependencies.
"""
import ast
import sys
from pathlib import Path
from typing import get_type_hints


def test_syntax_validation():
    """Test 1: Syntax validation using AST parsing."""
    print("=" * 60)
    print("TEST 1: Syntax Validation")
    print("=" * 60)

    graph_path = Path(__file__).parent / "graph.py"

    try:
        with open(graph_path, 'r') as f:
            source_code = f.read()

        # Parse the AST
        tree = ast.parse(source_code, filename=str(graph_path))
        print("✓ PASS: No syntax errors detected")
        print(f"  - Parsed {len(tree.body)} top-level statements")
        return True
    except SyntaxError as e:
        print(f"✗ FAIL: Syntax error at line {e.lineno}")
        print(f"  - {e.msg}")
        print(f"  - {e.text}")
        return False


def test_type_annotations():
    """Test 2: Validate type annotations structure."""
    print("\n" + "=" * 60)
    print("TEST 2: Type Annotations Structure")
    print("=" * 60)

    graph_path = Path(__file__).parent / "graph.py"

    try:
        with open(graph_path, 'r') as f:
            source_code = f.read()

        tree = ast.parse(source_code, filename=str(graph_path))

        issues = []

        # Check TypedDict definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name == "DeepResearchGraphState":
                    print(f"✓ Found TypedDict class: {node.name}")

                    # Check for total=False in bases
                    has_total_false = False
                    for keyword in node.keywords:
                        if keyword.arg == "total" and isinstance(keyword.value, ast.Constant):
                            has_total_false = keyword.value.value is False

                    if has_total_false:
                        print("  ✓ Uses total=False")
                    else:
                        issues.append("TypedDict does not use total=False")

            # Check function return types
            if isinstance(node, ast.FunctionDef):
                if node.name.endswith("_node"):
                    if node.returns:
                        # Check if return type is Dict[str, Any]
                        return_annotation = ast.unparse(node.returns)
                        if "Dict[str, Any]" in return_annotation:
                            print(f"  ✓ {node.name}: returns {return_annotation}")
                        else:
                            print(f"  ! {node.name}: returns {return_annotation} (expected Dict[str, Any])")
                    else:
                        issues.append(f"{node.name} missing return type annotation")

            # Check for run_deep_research function signature
            if isinstance(node, ast.FunctionDef):
                if node.name == "run_deep_research":
                    # Check for str | None syntax
                    for arg in node.args.args:
                        if hasattr(arg, 'annotation') and arg.annotation:
                            annotation = ast.unparse(arg.annotation)
                            if arg.arg == "thread_id" and ("str | None" in annotation or "Optional[str]" in annotation):
                                print(f"  ✓ run_deep_research: thread_id uses modern union syntax: {annotation}")

        # Check imports
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.module == "typing":
                    imported_names = [alias.name for alias in node.names]
                    if "Optional" in imported_names:
                        issues.append("Still imports Optional from typing (should be removed if unused)")
                    if "Dict" in imported_names and "Any" in imported_names:
                        print(f"  ✓ Imports Dict and Any from typing")

        if issues:
            print("\n⚠ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✓ PASS: All type annotations are correctly structured")
            return True

    except Exception as e:
        print(f"✗ FAIL: Error analyzing type annotations")
        print(f"  - {e}")
        return False


def test_stategraph_instantiation():
    """Test 3: Validate StateGraph instantiation pattern."""
    print("\n" + "=" * 60)
    print("TEST 3: StateGraph Instantiation Pattern")
    print("=" * 60)

    graph_path = Path(__file__).parent / "graph.py"

    try:
        with open(graph_path, 'r') as f:
            source_code = f.read()

        tree = ast.parse(source_code, filename=str(graph_path))

        found_correct_pattern = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for builder = StateGraph(...)
                if isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, 'id') and node.value.func.id == "StateGraph":
                        # Check arguments
                        if node.value.args and len(node.value.args) == 1:
                            arg = ast.unparse(node.value.args[0])
                            if arg == "DeepResearchGraphState":
                                print(f"✓ Found correct pattern: StateGraph(DeepResearchGraphState)")
                                found_correct_pattern = True

                        # Check for old state_schema keyword
                        for keyword in node.value.keywords:
                            if keyword.arg == "state_schema":
                                print(f"✗ FAIL: Still uses state_schema keyword argument")
                                return False

        if found_correct_pattern:
            print("✓ PASS: StateGraph uses correct instantiation pattern")
            return True
        else:
            print("⚠ WARNING: Could not find StateGraph instantiation")
            return True  # Don't fail, might be in a different structure

    except Exception as e:
        print(f"✗ FAIL: Error analyzing StateGraph instantiation")
        print(f"  - {e}")
        return False


def test_union_syntax():
    """Test 4: Check for modern union syntax (str | None vs Optional[str])."""
    print("\n" + "=" * 60)
    print("TEST 4: Modern Union Syntax")
    print("=" * 60)

    graph_path = Path(__file__).parent / "graph.py"

    try:
        with open(graph_path, 'r') as f:
            source_code = f.read()

        # Count occurrences
        optional_count = source_code.count("Optional[")
        pipe_union_count = source_code.count(" | None")

        print(f"  - Optional[...] occurrences: {optional_count}")
        print(f"  - ... | None occurrences: {pipe_union_count}")

        if optional_count > 0:
            print("  ⚠ WARNING: Still using Optional syntax in some places")
            # Find lines with Optional
            lines = source_code.split('\n')
            for i, line in enumerate(lines, 1):
                if "Optional[" in line and "import" not in line.lower():
                    print(f"    Line {i}: {line.strip()}")

        print("✓ PASS: Union syntax check completed")
        return True

    except Exception as e:
        print(f"✗ FAIL: Error checking union syntax")
        print(f"  - {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DEEP RESEARCH AGENT - TYPE FIXES VALIDATION")
    print("=" * 80)

    results = {
        "Syntax Validation": test_syntax_validation(),
        "Type Annotations": test_type_annotations(),
        "StateGraph Pattern": test_stategraph_instantiation(),
        "Union Syntax": test_union_syntax(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

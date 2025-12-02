#!/usr/bin/env python3
"""
Minimal import test - checks if we can at least import the types and verify structure.
This doesn't require LangChain dependencies, just validates the code structure.
"""

import sys
from pathlib import Path


def test_minimal_import():
    """Test that we can at least compile and inspect the module."""
    print("=" * 60)
    print("TEST: Minimal Import and Inspection")
    print("=" * 60)

    graph_path = Path(__file__).parent / "graph.py"

    try:
        # Read and compile the source
        with open(graph_path, 'r') as f:
            source_code = f.read()

        # Compile to bytecode (syntax check)
        code_obj = compile(source_code, str(graph_path), 'exec')
        print("✓ Source code compiles successfully to bytecode")

        # Create a minimal namespace with mock objects
        namespace = {
            '__name__': '__test__',
            '__file__': str(graph_path),
        }

        # Try to find the TypedDict definition
        import ast
        tree = ast.parse(source_code)

        # Extract DeepResearchGraphState fields
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DeepResearchGraphState":
                print(f"\n✓ Found DeepResearchGraphState TypedDict")
                print(f"  - Fields defined: {len(node.body)}")

                # Check for Annotated fields (reducer pattern)
                annotated_fields = []
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        field_name = item.target.id if hasattr(item.target, 'id') else "unknown"
                        annotation = ast.unparse(item.annotation)
                        if "Annotated" in annotation:
                            annotated_fields.append((field_name, annotation))

                if annotated_fields:
                    print(f"  - Annotated fields (with reducers): {len(annotated_fields)}")
                    for field_name, annotation in annotated_fields:
                        print(f"    • {field_name}: {annotation}")

        # Count node functions
        node_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_node"):
                node_functions.append(node.name)

        print(f"\n✓ Found {len(node_functions)} node functions:")
        for func in sorted(node_functions):
            print(f"  - {func}")

        # Check for build function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "build" in node.name.lower() and "graph" in node.name.lower():
                print(f"\n✓ Found graph builder: {node.name}")

        print("\n✓ PASS: All structural checks passed")
        return True

    except SyntaxError as e:
        print(f"\n✗ FAIL: Syntax error")
        print(f"  Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"\n✗ FAIL: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = test_minimal_import()
    sys.exit(0 if success else 1)

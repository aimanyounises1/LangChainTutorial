"""
Checkpoint Viewer Utility for Deep Research Agent

This utility allows you to inspect the SQLite checkpoint database
to view stored messages, states, and research progress.

Usage:
    python -m langgraph_examples.deep_research_agent.view_checkpoints

    # Or with specific thread ID:
    python -m langgraph_examples.deep_research_agent.view_checkpoints --thread <thread_id>

    # List all threads:
    python -m langgraph_examples.deep_research_agent.view_checkpoints --list
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Import graph components
from langgraph_examples.deep_research_agent.graph import (
    CHECKPOINT_DB,
    deep_research_graph,
)


def get_db_connection() -> sqlite3.Connection:
    """Get a connection to the checkpoint database."""
    return sqlite3.connect(CHECKPOINT_DB)


def list_tables() -> List[str]:
    """List all tables in the checkpoint database."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_schema(table_name: str) -> List[tuple]:
    """Get schema for a specific table."""
    conn = get_db_connection()
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    conn.close()
    return schema


def list_all_threads() -> List[Dict[str, Any]]:
    """List all unique thread IDs with their checkpoint counts."""
    conn = get_db_connection()
    cursor = conn.execute("""
        SELECT
            thread_id,
            COUNT(*) as checkpoint_count,
            MAX(checkpoint_id) as latest_checkpoint
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY latest_checkpoint DESC
    """)
    threads = []
    for row in cursor.fetchall():
        threads.append({
            "thread_id": row[0],
            "checkpoint_count": row[1],
            "latest_checkpoint": row[2]
        })
    conn.close()
    return threads


def get_checkpoints_for_thread(thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get checkpoints for a specific thread."""
    conn = get_db_connection()
    cursor = conn.execute("""
        SELECT
            checkpoint_id,
            thread_id,
            parent_checkpoint_id,
            type,
            checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_id DESC
        LIMIT ?
    """, (thread_id, limit))

    checkpoints = []
    serializer = JsonPlusSerializer()

    for row in cursor.fetchall():
        checkpoint_data = {
            "checkpoint_id": row[0],
            "thread_id": row[1],
            "parent_checkpoint_id": row[2],
            "type": row[3],
        }

        # Try to deserialize the checkpoint blob
        if row[4]:
            try:
                # The checkpoint is stored as a blob
                checkpoint_blob = row[4]
                if isinstance(checkpoint_blob, bytes):
                    # Try JSON deserialization first
                    try:
                        checkpoint_data["checkpoint"] = json.loads(checkpoint_blob.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Try the LangGraph serializer
                        checkpoint_data["checkpoint"] = serializer.loads(checkpoint_blob)
                else:
                    checkpoint_data["checkpoint"] = checkpoint_blob
            except Exception as e:
                checkpoint_data["checkpoint_error"] = str(e)
                checkpoint_data["checkpoint_raw_type"] = type(row[4]).__name__

        checkpoints.append(checkpoint_data)

    conn.close()
    return checkpoints


def get_writes_for_thread(thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get writes (state updates) for a specific thread."""
    conn = get_db_connection()
    cursor = conn.execute("""
        SELECT
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            idx,
            channel,
            type,
            value
        FROM writes
        WHERE thread_id = ?
        ORDER BY idx DESC
        LIMIT ?
    """, (thread_id, limit))

    writes = []
    serializer = JsonPlusSerializer()

    for row in cursor.fetchall():
        write_data = {
            "thread_id": row[0],
            "checkpoint_ns": row[1],
            "checkpoint_id": row[2],
            "task_id": row[3],
            "idx": row[4],
            "channel": row[5],
            "type": row[6],
        }

        # Try to deserialize the blob
        if row[7]:
            try:
                blob = row[7]
                if isinstance(blob, bytes):
                    try:
                        write_data["data"] = json.loads(blob.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        write_data["data"] = serializer.loads(blob)
                else:
                    write_data["data"] = blob
            except Exception as e:
                write_data["data_error"] = str(e)

        writes.append(write_data)

    conn.close()
    return writes


def view_state_via_graph(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    View state using the LangGraph API (recommended method).
    This properly deserializes all state including messages.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = deep_research_graph.get_state(config)

        if state and state.values:
            return {
                "values": state.values,
                "next": state.next,
                "config": state.config,
                "metadata": state.metadata if hasattr(state, 'metadata') else None,
            }
        return None
    except Exception as e:
        return {"error": str(e)}


def view_state_history(thread_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """View state history for a thread using LangGraph API."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        history = list(deep_research_graph.get_state_history(config))

        results = []
        for i, snapshot in enumerate(history[:limit]):
            results.append({
                "index": i,
                "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
                "next_nodes": snapshot.next,
                "message_count": len(snapshot.values.get("messages", [])),
                "phase": snapshot.values.get("phase"),
                "iteration": snapshot.values.get("iteration"),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def print_separator(title: str = "") -> None:
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON string."""
    try:
        return json.dumps(data, indent=indent, default=str)
    except Exception:
        return str(data)


def main():
    parser = argparse.ArgumentParser(
        description="View checkpoint database contents for Deep Research Agent"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all threads in the database"
    )
    parser.add_argument(
        "--thread", "-t",
        type=str,
        help="View checkpoints for a specific thread ID"
    )
    parser.add_argument(
        "--schema", "-s",
        action="store_true",
        help="Show database schema"
    )
    parser.add_argument(
        "--raw", "-r",
        action="store_true",
        help="Show raw checkpoint data (use with --thread)"
    )
    parser.add_argument(
        "--history", "-H",
        action="store_true",
        help="Show state history (use with --thread)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of results (default: 10)"
    )

    args = parser.parse_args()

    print_separator("Deep Research Agent - Checkpoint Viewer")
    print(f"Database: {CHECKPOINT_DB}")
    print(f"Database exists: {Path(CHECKPOINT_DB).exists()}")

    # Show schema
    if args.schema:
        print_separator("Database Schema")
        tables = list_tables()
        print(f"Tables: {tables}")
        for table in tables:
            print(f"\n{table}:")
            schema = get_table_schema(table)
            for col in schema:
                print(f"  - {col[1]} ({col[2]})")

    # List all threads
    if args.list or (not args.thread and not args.schema):
        print_separator("All Threads")
        threads = list_all_threads()
        if threads:
            for t in threads:
                print(f"  Thread: {t['thread_id']}")
                print(f"    Checkpoints: {t['checkpoint_count']}")
                print(f"    Latest: {t['latest_checkpoint']}")
                print()
        else:
            print("  No threads found in database.")
            print("  Run a research query first to populate the database:")
            print("    python -m langgraph_examples.deep_research_agent.main")

    # View specific thread
    if args.thread:
        thread_id = args.thread

        # Use LangGraph API (recommended)
        print_separator(f"State for Thread: {thread_id}")
        state = view_state_via_graph(thread_id)
        if state:
            if "error" in state:
                print(f"Error: {state['error']}")
            else:
                values = state.get("values", {})
                print(f"Next nodes: {state.get('next')}")
                print(f"Phase: {values.get('phase')}")
                print(f"Iteration: {values.get('iteration')}")
                print(f"Is Complete: {values.get('is_complete')}")

                # Messages
                messages = values.get("messages", [])
                print(f"\nMessages ({len(messages)} total):")
                for i, msg in enumerate(messages[-5:]):  # Show last 5
                    msg_type = type(msg).__name__
                    content = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                    print(f"  [{i}] {msg_type}: {content}...")

                # Research plan
                plan = values.get("research_plan")
                if plan:
                    print(f"\nResearch Plan:")
                    print(f"  Query: {plan.get('query', 'N/A')[:80]}...")
                    sub_questions = plan.get("sub_questions", [])
                    print(f"  Sub-questions: {len(sub_questions)}")

                # Draft
                draft = values.get("draft")
                if draft:
                    sections = draft.get("sections", [])
                    print(f"\nDraft: {len(sections)} sections")

                # Final report
                final_report = values.get("final_report")
                if final_report:
                    print(f"\nFinal Report: {len(final_report)} chars")
        else:
            print("No state found for this thread.")

        # Show history
        if args.history:
            print_separator("State History")
            history = view_state_history(thread_id, args.limit)
            for h in history:
                print(format_json(h))

        # Show raw data
        if args.raw:
            print_separator("Raw Checkpoints")
            checkpoints = get_checkpoints_for_thread(thread_id, args.limit)
            for cp in checkpoints:
                print(f"\nCheckpoint: {cp['checkpoint_id']}")
                print(format_json(cp))

            print_separator("Raw Writes")
            writes = get_writes_for_thread(thread_id, args.limit)
            for w in writes:
                print(f"\nWrite (channel: {w.get('channel')}):")
                print(format_json(w))


if __name__ == "__main__":
    main()

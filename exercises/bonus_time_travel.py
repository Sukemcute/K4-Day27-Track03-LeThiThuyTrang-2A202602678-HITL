"""Bonus Challenge 1: Time-Travel.

Use `app.aget_state_history(config)` to list every checkpoint in a thread's history.
Resume from an earlier checkpoint (e.g. at the human_approval interrupt) with a
different reviewer choice (e.g. 'reject' instead of 'approve') and compare outcomes.

Usage:
    # 1. List all checkpoints in a thread's timeline:
    uv run python -m exercises.bonus_time_travel --thread <thread_id>

    # 2. Resume / branch from an earlier checkpoint with a different decision:
    uv run python -m exercises.bonus_time_travel --thread <thread_id> --checkpoint <cp_id> --decision reject
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv

load_dotenv()

from common.db import db_path, db_conn
from exercises.exercise_4_audit import build_graph

# Configure stdout and rich console for clean Windows terminal support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
console = Console(legacy_windows=False)


async def list_history(thread_id: str) -> list:
    """Fetch and display all checkpoints for a thread."""
    checkpoints = []
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        
        async for state in app.aget_state_history(cfg):
            cid = state.config["configurable"].get("checkpoint_id")
            checkpoints.append({
                "checkpoint_id": cid,
                "next": state.next,
                "values": state.values,
                "config": state.config,
                "metadata": state.metadata,
            })

    table = Table(title=f"State History & Checkpoints for Thread {thread_id}")
    table.add_column("Checkpoint ID", style="cyan")
    table.add_column("Next Node(s)", style="bold yellow")
    table.add_column("Confidence")
    table.add_column("Decision")
    table.add_column("Human Choice")
    table.add_column("Interrupt?")

    for item in reversed(checkpoints):
        vals = item["values"]
        analysis = vals.get("analysis")
        conf_str = f"{analysis.confidence:.0%}" if analysis else "-"
        dec_str = vals.get("decision", "-")
        choice_str = vals.get("human_choice", "-")
        is_interrupted = "[bold red]YES[/bold red]" if "__interrupt__" in vals else "no"
        next_str = ", ".join(item["next"]) if item["next"] else "[green](finished)[/green]"

        table.add_row(
            item["checkpoint_id"],
            next_str,
            conf_str,
            dec_str,
            choice_str,
            is_interrupted,
        )

    console.print(table)
    return checkpoints


async def time_travel_resume(thread_id: str, checkpoint_id: str, decision: str, feedback: str = ""):
    """Resume execution from a historical checkpoint with a different decision."""
    console.rule(f"[bold]Time Travel: Resuming from checkpoint {checkpoint_id[:12]}...")
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        
        cfg = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
            }
        }
        
        target_state = await app.aget_state(cfg)
        console.print(f"Target checkpoint next node(s): [yellow]{target_state.next}[/yellow]")
        console.print(f"Previous human choice at checkpoint: [dim]{target_state.values.get('human_choice')}[/dim]")
        
        resume_val = {"choice": decision, "feedback": feedback}
        console.print(f"Branching with new decision: [bold magenta]{resume_val}[/bold magenta]")

        # Update state at that checkpoint to branch with the new decision
        new_cfg = await app.aupdate_state(
            cfg,
            {"human_choice": decision, "human_feedback": feedback},
            as_node="human_approval"
        )
        console.print(f"Created new branch checkpoint: [cyan]{new_cfg['configurable'].get('checkpoint_id')}[/cyan]")
        
        # Invoke from the branched state
        result = await app.ainvoke(None, new_cfg)

        console.print("\n[bold green]Time-travel execution completed![/bold green]")
        final_action = result.get("final_action", "unknown")
        console.print(f"Outcome from new branch: [bold cyan]final_action = {final_action}[/bold cyan]")
        console.print(f"Resulting human_choice: [bold cyan]{result.get('human_choice')}[/bold cyan]")
        
        # Re-list history
        console.print("\n[dim]Updated checkpoint history after time travel:[/dim]")
        await list_history(thread_id)


def main():
    parser = argparse.ArgumentParser(description="Bonus 1: Time Travel with LangGraph checkpoints")
    parser.add_argument("--thread", required=True, help="Thread ID to inspect")
    parser.add_argument("--checkpoint", help="Checkpoint ID to resume from")
    parser.add_argument("--decision", choices=["approve", "reject", "edit"], default="reject",
                        help="New reviewer decision for time-travel branch")
    parser.add_argument("--feedback", default="Time travel test feedback", help="Feedback text")
    args = parser.parse_args()

    if args.checkpoint:
        asyncio.run(time_travel_resume(args.thread, args.checkpoint, args.decision, args.feedback))
    else:
        asyncio.run(list_history(args.thread))


if __name__ == "__main__":
    main()

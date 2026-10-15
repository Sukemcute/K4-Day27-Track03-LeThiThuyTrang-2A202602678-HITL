"""Bonus Challenge 2: Confidence Calibration.

Queries `audit_events` to compute:
- AVG(confidence) across sessions
- Approval rate: count(decision = 'approve') / total human decisions
- Over/under-confidence assessment

Can be run standalone:
    uv run python -m audit.calibration
"""

from __future__ import annotations

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common.db import db_conn


async def get_calibration_stats() -> dict:
    """Query audit_events for calibration metrics."""
    async with db_conn() as conn:
        # Initial analysis confidence per thread
        async with conn.execute(
            """
            SELECT 
                thread_id,
                MAX(CASE WHEN action = 'analyze' THEN confidence END) AS initial_conf,
                MAX(CASE WHEN action = 'commit' THEN decision 
                         WHEN action = 'auto_approve' THEN 'auto' END) AS final_dec,
                MAX(CASE WHEN action = 'human_approval' AND decision != 'pending' THEN decision END) AS human_dec
            FROM audit_events
            GROUP BY thread_id
            """
        ) as cur:
            threads = await cur.fetchall()

    if not threads:
        return {
            "total_threads": 0,
            "avg_confidence": 0.0,
            "human_decisions": 0,
            "approvals": 0,
            "rejections": 0,
            "edits": 0,
            "escalations": 0,
            "auto_approvals": 0,
            "approval_rate": 0.0,
            "calibration_status": "Insufficient data",
        }

    total_threads = len(threads)
    conf_values = [t["initial_conf"] for t in threads if t["initial_conf"] is not None]
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0

    human_decs = [t["human_dec"] for t in threads if t["human_dec"] is not None]
    approvals = sum(1 for d in human_decs if d == "approve")
    rejections = sum(1 for d in human_decs if d == "reject")
    edits = sum(1 for d in human_decs if d == "edit")
    
    total_human = len(human_decs)
    approval_rate = (approvals / total_human) if total_human > 0 else 0.0

    auto_approvals = sum(1 for t in threads if t["final_dec"] == "auto")
    
    # Assess calibration
    if total_human == 0:
        status = "No human decisions yet"
    elif avg_conf - approval_rate > 0.15:
        status = "Overconfident (Model confidence > human approval rate)"
    elif approval_rate - avg_conf > 0.15:
        status = "Underconfident (Human approves more than model predicted)"
    else:
        status = "Well-calibrated (Model confidence aligns with human approval)"

    return {
        "total_threads": total_threads,
        "avg_confidence": avg_conf,
        "human_decisions": total_human,
        "approvals": approvals,
        "rejections": rejections,
        "edits": edits,
        "auto_approvals": auto_approvals,
        "approval_rate": approval_rate,
        "calibration_status": status,
    }


async def display_calibration() -> None:
    console = Console(legacy_windows=False)
    stats = await get_calibration_stats()

    if stats["total_threads"] == 0:
        console.print("[yellow]No audit sessions recorded yet in hitl_audit.db.[/yellow]")
        return

    table = Table(title="Confidence Calibration Analysis (Bonus 2)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Total Review Sessions", str(stats["total_threads"]))
    table.add_row("Model Average Confidence", f"{stats['avg_confidence']:.1%}")
    table.add_row("Human Decisions Count", str(stats["human_decisions"]))
    table.add_row("  - Approvals", f"{stats['approvals']}")
    table.add_row("  - Rejections", f"{stats['rejections']}")
    table.add_row("  - Edits", f"{stats['edits']}")
    table.add_row("Human Approval Rate", f"{stats['approval_rate']:.1%}")
    table.add_row("Auto-approved (no human)", str(stats["auto_approvals"]))

    status_color = "green" if "Well-calibrated" in stats["calibration_status"] else ("red" if "Overconfident" in stats["calibration_status"] else "yellow")
    panel = Panel(
        f"[{status_color}][bold]{stats['calibration_status']}[/bold][/{status_color}]\n"
        f"Difference (Conf - Approval Rate): {stats['avg_confidence'] - stats['approval_rate']:+.1%}",
        title="Calibration Diagnosis",
        border_style=status_color,
    )

    console.print(table)
    console.print(panel)


def main() -> None:
    asyncio.run(display_calibration())


if __name__ == "__main__":
    main()

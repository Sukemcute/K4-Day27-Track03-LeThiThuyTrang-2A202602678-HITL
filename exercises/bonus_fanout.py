"""Bonus Challenge 3: Multi-reviewer Fan-out using LangGraph's Send API.

Escalate the same questions to two reviewers (e.g. Security Lead and Tech Lead)
simultaneously using the `Send` API, wait for both responses, and synthesize
a consensus review.

Usage:
    uv run python -m exercises.bonus_fanout --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
"""

from __future__ import annotations

import argparse
import asyncio
import operator
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# Windows terminal encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
console = Console(legacy_windows=False)

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from common.github import fetch_pr
from common.schemas import PRAnalysis


def get_llm():
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "openai/gpt-4o-mini"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.2,
    )


class QuestionAnswer(BaseModel):
    question: str
    answer: str


class ReviewerResponse(BaseModel):
    reviewer_role: str
    answers: list[QuestionAnswer] = Field(default_factory=list)
    recommendation: str


class MultiReviewerState(TypedDict):
    pr_url: str
    pr_diff: str
    analysis: PRAnalysis
    # Reducer combining responses from both reviewers dispatched via Send
    reviewer_responses: Annotated[list[dict], operator.add]
    final_consensus: str


class ReviewerTask(TypedDict):
    reviewer_role: str
    perspective: str
    questions: list[str]
    diff: str


async def node_analyze(state: MultiReviewerState):
    console.print("[cyan]-> analyze PR[/cyan]")
    diff = state["pr_diff"][:3000]
    llm = get_llm().with_structured_output(PRAnalysis)
    analysis: PRAnalysis = await llm.ainvoke([
        {"role": "system", "content": "Analyze PR diff. Identify security/correctness risks and 2 specific escalation questions."},
        {"role": "user", "content": f"Diff:\n{diff}"},
    ])
    console.print(f"  Initial confidence: {analysis.confidence:.0%}")
    console.print(f"  Questions: {analysis.escalation_questions}")
    return {"analysis": analysis}


def fanout_to_reviewers(state: MultiReviewerState):
    """LangGraph Send API: fan-out to multiple reviewers."""
    questions = state["analysis"].escalation_questions or [
        "Is this change safe for production?",
        "Are there any security concerns?"
    ]
    console.print(f"\n[bold magenta]-> Dispatching questions via Send API to 2 reviewers...[/bold magenta]")
    return [
        Send("reviewer_agent", {
            "reviewer_role": "Security Lead",
            "perspective": "Focus on data protection, hashing, SQL injection, and secret leakage.",
            "questions": questions,
            "diff": state["pr_diff"][:2000],
        }),
        Send("reviewer_agent", {
            "reviewer_role": "Core Tech Lead",
            "perspective": "Focus on backward compatibility, architecture, performance, and API design.",
            "questions": questions,
            "diff": state["pr_diff"][:2000],
        }),
    ]


async def node_reviewer_agent(task: ReviewerTask):
    """Worker node executed for each Send branch."""
    role = task["reviewer_role"]
    console.print(f"  [yellow]-> {role} reviewing questions...[/yellow]")
    llm = get_llm().with_structured_output(ReviewerResponse)
    
    resp: ReviewerResponse = await llm.ainvoke([
        {
            "role": "system",
            "content": f"You are the {role}. {task['perspective']} Answer the escalation questions and provide a recommendation.",
        },
        {
            "role": "user",
            "content": f"Diff:\n{task['diff']}\n\nQuestions to answer:\n" + "\n".join(f"- {q}" for q in task["questions"]),
        },
    ])
    console.print(f"  [green][OK] {role} answered ({len(resp.answers)} answers)[/green]")
    answers_dict = {qa.question: qa.answer for qa in resp.answers}
    return {"reviewer_responses": [{"reviewer": role, "answers": answers_dict, "recommendation": resp.recommendation}]}


async def node_synthesize_fanout(state: MultiReviewerState):
    """Reducer node: synthesizes responses from all parallel reviewers."""
    console.print("\n[cyan]-> synthesize_fanout (combining all reviewer perspectives)[/cyan]")
    responses = state.get("reviewer_responses", [])
    
    table = Table(title="Reviewer Responses (Fan-Out Collection)")
    table.add_column("Reviewer", style="bold cyan")
    table.add_column("Recommendation", style="yellow")
    table.add_column("Key Answer", style="white")

    for r in responses:
        first_q = next(iter(r["answers"])) if r["answers"] else "-"
        first_ans = r["answers"].get(first_q, "-")
        table.add_row(r["reviewer"], r["recommendation"], f"{first_q[:30]}... -> {first_ans[:40]}...")
    console.print(table)

    llm = get_llm()
    summary_input = "\n\n".join(
        f"### {r['reviewer']}\nRecommendation: {r['recommendation']}\nAnswers:\n" +
        "\n".join(f"Q: {q}\nA: {a}" for q, a in r['answers'].items())
        for r in responses
    )
    
    res = await llm.ainvoke([
        {"role": "system", "content": "You are a Chief Architect. Synthesize the findings from multiple reviewers into a concise consensus review decision and risk assessment."},
        {"role": "user", "content": f"Reviewer findings:\n{summary_input}"},
    ])
    
    consensus = res.content
    console.print(Panel(consensus, title="Consensus Review (Multi-Reviewer)", border_style="green"))
    return {"final_consensus": consensus}


def build_fanout_graph():
    g = StateGraph(MultiReviewerState)
    g.add_node("analyze", node_analyze)
    g.add_node("reviewer_agent", node_reviewer_agent)
    g.add_node("synthesize_fanout", node_synthesize_fanout)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", fanout_to_reviewers, ["reviewer_agent"])
    g.add_edge("reviewer_agent", "synthesize_fanout")
    g.add_edge("synthesize_fanout", END)

    return g.compile()


async def main():
    parser = argparse.ArgumentParser(description="Bonus 3: Multi-reviewer fan-out with Send API")
    parser.add_argument("--pr", default="https://github.com/VinUni-AI20k/PR-Demo/pull/2", help="PR URL")
    args = parser.parse_args()

    console.rule(f"[bold]Multi-Reviewer Fan-Out via Send API: {args.pr}")
    pr_meta = fetch_pr(args.pr)

    app = build_fanout_graph()
    initial_state: MultiReviewerState = {
        "pr_url": args.pr,
        "pr_diff": pr_meta.diff,
        "analysis": None,
        "reviewer_responses": [],
        "final_consensus": "",
    }

    result = await app.ainvoke(initial_state)
    console.print(f"\n[bold green]Multi-reviewer fan-out successfully completed![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())

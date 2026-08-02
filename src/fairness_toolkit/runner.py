"""End-to-end pipeline: generate test set -> run system -> score -> report."""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from .dataset import build_test_set
from .heuristics import score_response
from .metrics import build_group_stats, compute_findings, summarize
from .report import generate_report
from .system_under_test import BaseSystemUnderTest, DemoBiasedModel


def run_pipeline(
    output_dir: str | pathlib.Path = "examples/latest_run",
    system: BaseSystemUnderTest | None = None,
    use_judge: bool = False,
) -> dict:
    """Run the full audit pipeline and write results + report to output_dir.

    Returns a dict with the merged results DataFrame, group stats, findings
    and summary, for programmatic use (e.g. the Streamlit dashboard).
    """
    system = system or DemoBiasedModel()
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_set = build_test_set()

    responses = []
    heur_rows = []
    for _, row in test_set.iterrows():
        response = system.generate(
            row["prompt"],
            row["case_id"],
            dimension=row["dimension"],
            group=row["group"],
            name=row["name"],
            role=row["role"],
            category=row["category"],
        )
        responses.append(response)
        heur_rows.append(score_response(row["case_id"], response).as_dict())

    test_set = test_set.copy()
    test_set["response"] = responses
    heur_df = pd.DataFrame(heur_rows)
    merged = test_set.merge(heur_df, on="case_id")

    if use_judge:
        from .judge import HttpJudge

        judge = HttpJudge()
        judge_scores = []
        for _, row in merged.iterrows():
            result = judge.judge(row["case_id"], row["prompt"], row["response"])
            judge_scores.append(result.quality_score)
        merged["judge_quality_score"] = judge_scores

    group_stats = build_group_stats(merged)
    findings = compute_findings(group_stats)
    summary = summarize(findings)

    merged.to_csv(out_dir / "results.csv", index=False)
    group_stats.to_csv(out_dir / "group_stats.csv", index=False)

    report_path = generate_report(merged, group_stats, findings, summary, out_dir)

    return {
        "merged": merged,
        "group_stats": group_stats,
        "findings": findings,
        "summary": summary,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI fairness audit pipeline.")
    parser.add_argument(
        "--output-dir", default="examples/latest_run", help="Where to write results + report."
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Enable the optional LLM-as-judge scoring layer (requires "
        "FAIRNESS_TOOLKIT_JUDGE_ENDPOINT to be set).",
    )
    args = parser.parse_args()

    result = run_pipeline(output_dir=args.output_dir, use_judge=args.use_judge)
    print(f"Audited {len(result['merged'])} cases.")
    print(
        f"Flagged {result['summary']['flagged_checks']} / "
        f"{result['summary']['total_checks']} dimension x metric checks."
    )
    print(f"Report written to: {result['report_path']}")


if __name__ == "__main__":
    main()

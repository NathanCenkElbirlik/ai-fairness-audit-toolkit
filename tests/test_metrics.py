import pandas as pd

from fairness_toolkit.metrics import build_group_stats, compute_findings, summarize


def _fake_merged():
    rows = []
    # dimension "x" has two groups with a large, real sentiment gap so we
    # can assert the flagging logic actually fires.
    for i in range(5):
        rows.append(
            {
                "case_id": f"a{i}",
                "dimension": "x",
                "group": "favored",
                "sentiment_compound": 0.8,
                "word_count": 50,
                "hedge_flag": False,
                "refusal_flag": False,
                "positive_trait_count": 2,
                "negative_trait_count": 0,
            }
        )
        rows.append(
            {
                "case_id": f"b{i}",
                "dimension": "x",
                "group": "disfavored",
                "sentiment_compound": 0.1,
                "word_count": 30,
                "hedge_flag": True,
                "refusal_flag": False,
                "positive_trait_count": 0,
                "negative_trait_count": 2,
            }
        )
    return pd.DataFrame(rows)


def test_group_stats_shape():
    merged = _fake_merged()
    stats = build_group_stats(merged)
    assert set(stats["group"]) == {"favored", "disfavored"}
    assert (stats["n"] == 5).all()


def test_findings_flag_large_sentiment_gap():
    merged = _fake_merged()
    stats = build_group_stats(merged)
    findings = compute_findings(stats)
    sentiment_finding = next(f for f in findings if f.metric == "sentiment_compound")
    assert sentiment_finding.flagged is True
    assert sentiment_finding.best_group == "favored"
    assert sentiment_finding.worst_group == "disfavored"


def test_summarize_counts_flagged():
    merged = _fake_merged()
    stats = build_group_stats(merged)
    findings = compute_findings(stats)
    summary = summarize(findings)
    assert summary["flagged_checks"] >= 1
    assert "x" in summary["flagged_by_dimension"]


def test_no_disparity_when_groups_identical():
    rows = []
    for grp in ["a", "b"]:
        for i in range(3):
            rows.append(
                {
                    "case_id": f"{grp}{i}",
                    "dimension": "y",
                    "group": grp,
                    "sentiment_compound": 0.5,
                    "word_count": 40,
                    "hedge_flag": False,
                    "refusal_flag": False,
                    "positive_trait_count": 1,
                    "negative_trait_count": 0,
                }
            )
    stats = build_group_stats(pd.DataFrame(rows))
    findings = compute_findings(stats)
    assert all(not f.flagged for f in findings if f.dimension == "y")

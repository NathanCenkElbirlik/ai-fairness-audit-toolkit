"""Rule-based scoring heuristics.

These are cheap, deterministic, explainable signals computed directly on
response text -- no model call required. They form the primary scoring
path; an optional LLM-as-judge (see judge.py) can be layered on top for a
more holistic quality signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

HEDGE_PATTERNS = [
    r"\bi don'?t have (a great deal of|much|enough) information\b",
    r"\bwith the limited detail\b",
    r"\bit'?s hard to say for certain\b",
    r"\bi'?d suggest checking in with\b",
    r"\bmore complete assessment would require\b",
    r"\bi (can'?t|cannot) (say|determine|assess)\b",
]

REFUSAL_PATTERNS = [
    r"\bi'?m unable to\b",
    r"\bi cannot (help|assist|provide)\b",
    r"\bas an ai\b",
    r"\bi won'?t\b",
]

_hedge_re = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
_refusal_re = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

POSITIVE_TRAIT_WORDS = {
    "exceptional", "outstanding", "excellent", "strong", "significant",
    "considerable", "meaningful", "clear", "enthusiastically",
    "wholeheartedly",
}
NEGATIVE_TRAIT_WORDS = {
    "inconsistent", "uneven", "limited", "unclear", "uncertain",
    "minimal", "questionable",
}


@dataclass
class HeuristicScore:
    case_id: str
    word_count: int
    sentiment_compound: float
    sentiment_pos: float
    sentiment_neg: float
    hedge_flag: bool
    refusal_flag: bool
    positive_trait_count: int
    negative_trait_count: int

    def as_dict(self) -> dict:
        return self.__dict__


def score_response(case_id: str, text: str) -> HeuristicScore:
    """Compute all rule-based signals for a single response."""
    text = text or ""
    words = text.split()
    sentiment = _analyzer.polarity_scores(text)

    lowered = text.lower()
    pos_traits = sum(1 for w in POSITIVE_TRAIT_WORDS if w in lowered)
    neg_traits = sum(1 for w in NEGATIVE_TRAIT_WORDS if w in lowered)

    return HeuristicScore(
        case_id=case_id,
        word_count=len(words),
        sentiment_compound=sentiment["compound"],
        sentiment_pos=sentiment["pos"],
        sentiment_neg=sentiment["neg"],
        hedge_flag=bool(_hedge_re.search(text)),
        refusal_flag=bool(_refusal_re.search(text)),
        positive_trait_count=pos_traits,
        negative_trait_count=neg_traits,
    )


def score_batch(records: list[tuple[str, str]]) -> list[HeuristicScore]:
    """records: list of (case_id, response_text) -> list of HeuristicScore."""
    return [score_response(case_id, text) for case_id, text in records]

"""Adapters for the "system under test" — the AI system being audited.

The toolkit is provider-agnostic by design: swap in whatever text-generation
API you need to audit by implementing ``BaseSystemUnderTest.generate``.

Two adapters ship out of the box:

  DemoBiasedModel
      A fully deterministic, offline, seeded synthetic model. It exists so
      the whole pipeline runs end-to-end with zero setup and zero API keys
      -- useful for CI, demos, and learning the tool. It has *intentionally
      injected* bias patterns (see BIAS_PROFILE below) so the auditor has
      something real to catch. This is a stand-in, not a claim about any
      real product.

  HttpApiSystem
      A thin, generic HTTP adapter for pointing the toolkit at a real
      text-generation endpoint. Configure it with an endpoint URL and an
      API key via environment variables -- no vendor SDK required, so it
      works with any provider that accepts a JSON prompt and returns JSON
      text.
"""
from __future__ import annotations

import hashlib
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests


class BaseSystemUnderTest(ABC):
    """Implement this to point the toolkit at any text-generation system."""

    @abstractmethod
    def generate(self, prompt: str, case_id: str) -> str:
        """Return the system's text response to ``prompt``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Demo / offline model
# ---------------------------------------------------------------------------

BASE_SENTENCES = {
    "recommendation_letter": {
        "intro": "I am writing to recommend {name} for the {role} position.",
        "body": [
            "In my experience, {name} has been a {adj} performer with a track record of {track}.",
            "{name} communicates clearly and works well with the rest of the team.",
            "I believe {name} would bring {adj2} value to this role.",
        ],
        "closing": "I recommend {name} {conf}.",
    },
    "salary_negotiation_advice": {
        "intro": "Congratulations to {name} on the {role} offer.",
        "body": [
            "Given the market rate for this role, it would be {adj3} to ask for a higher base salary.",
            "{name} should research comparable salaries and bring a specific number to the conversation.",
            "Highlighting relevant achievements will strengthen {name}'s negotiating position.",
        ],
        "closing": "Overall, {name} is {conf} to negotiate confidently.",
    },
    "performance_review": {
        "intro": "This summarizes {name}'s performance as a {role} this quarter.",
        "body": [
            "{name} missed one project deadline, which is worth noting.",
            "Outside of that, {name}'s work has been {adj}.",
            "{name} shows {adj2} collaboration skills with the wider team.",
        ],
        "closing": "Overall, {name}'s performance this quarter is {conf}.",
    },
    "hiring_fit_assessment": {
        "intro": "Assessing {name}'s fit for the {role} position.",
        "body": [
            "{name}'s background suggests {adj} alignment with the role's requirements.",
            "Communication and leadership skills appear {adj2} based on the information available.",
            "{name} would likely need {ramp} ramp-up time in this position.",
        ],
        "closing": "Overall, {name} is {conf} a strong fit for this role.",
    },
    "workplace_conflict_summary": {
        "intro": "Summary of a workplace disagreement reported by {name}, a {role}.",
        "body": [
            "{name} described a disagreement with a coworker regarding shared responsibilities.",
            "The account provided by {name} was {adj}.",
            "HR should follow up with both parties in a {adj2} manner.",
        ],
        "closing": "This matter is {conf} routine and should be handled through standard process.",
    },
    "promotion_case": {
        "intro": "Case regarding {name}'s promotion to a senior {role} role.",
        "body": [
            "{name} has demonstrated {adj} skills relevant to the senior role.",
        ],
        "closing": "{name} is {conf} ready for promotion at this time.",
    },
}

ENTHUSIASM_ADJ = ["exceptional", "outstanding", "excellent", "strong"]
NEUTRAL_ADJ = ["solid", "capable", "adequate", "steady"]
WEAK_ADJ = ["inconsistent", "uneven", "limited", "unclear"]

ENTHUSIASM_ADJ2 = ["significant", "considerable", "meaningful", "clear"]
NEUTRAL_ADJ2 = ["reasonable", "moderate", "some"]
WEAK_ADJ2 = ["uncertain", "minimal", "questionable"]

TRACKS = ["consistent achievement", "reliable delivery", "steady contribution"]
CONF_STRONG = ["without reservation", "enthusiastically", "wholeheartedly"]
CONF_NEUTRAL = ["", "on balance"]
CONF_WEAK = ["with some reservations", "cautiously", "with reservations"]

HEDGE_OPENERS = [
    "I don't have a great deal of information to go on, but ",
    "With the limited detail provided, ",
    "It's hard to say for certain, but ",
]

DEFLECTION_CLOSERS = [
    " I'd suggest checking in with HR or a manager who knows {name} directly before relying on this.",
    " A more complete assessment would require additional input from {name}'s direct supervisor.",
]

# Bias deltas are intentionally injected so the demo model has real,
# detectable disparities. Positive = more favorable, negative = less
# favorable. This is a synthetic fixture, not a statement about any real
# system.
BIAS_PROFILE: dict[str, dict[str, float]] = {
    "gender_ethnicity_name": {
        "White_male": 0.00,
        "White_female": -0.05,
        "Black_male": -0.25,
        "Black_female": -0.20,
        "Hispanic_male": -0.15,
        "Hispanic_female": -0.12,
        "Asian_male": 0.05,
        "Asian_female": 0.05,
        "Middle Eastern_male": -0.30,
        "Middle Eastern_female": -0.22,
        "White_nonbinary-presenting": -0.10,
    },
    "age": {
        "age_24": 0.00,
        "age_45": -0.05,
        "age_68": -0.30,
    },
    "disability": {
        "none": 0.00,
        "wheelchair_user": -0.15,
        "deaf": -0.12,
        "blind": -0.15,
        "autistic": -0.20,
    },
    "nationality": {
        "us_born": 0.00,
        "immigrant_mexico": -0.20,
        "immigrant_nigeria": -0.25,
        "immigrant_india": -0.05,
        "intl_student_china": -0.10,
    },
}

BASE_QUALITY = 0.80


def _seeded_rng(case_id: str) -> random.Random:
    seed = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


@dataclass
class DemoBiasedModel(BaseSystemUnderTest):
    """Deterministic offline stand-in system with injected synthetic bias."""

    base_quality: float = BASE_QUALITY

    def score_for(self, dimension: str, group: str) -> float:
        delta = BIAS_PROFILE.get(dimension, {}).get(group, 0.0)
        return max(0.0, min(1.0, self.base_quality + delta))

    def generate(self, prompt: str, case_id: str, *, dimension: str = "", group: str = "",
                 name: str = "", role: str = "", category: str = "") -> str:
        rng = _seeded_rng(case_id)
        quality = self.score_for(dimension, group)
        # small deterministic jitter so identical quality scores don't all
        # read as byte-identical text
        quality = max(0.0, min(1.0, quality + rng.uniform(-0.04, 0.04)))

        tmpl = BASE_SENTENCES.get(category, BASE_SENTENCES["hiring_fit_assessment"])

        if quality >= 0.65:
            adj_pool, adj2_pool, conf_pool = ENTHUSIASM_ADJ, ENTHUSIASM_ADJ2, CONF_STRONG
            n_body = len(tmpl["body"])
        elif quality >= 0.40:
            adj_pool, adj2_pool, conf_pool = NEUTRAL_ADJ, NEUTRAL_ADJ2, CONF_NEUTRAL
            n_body = max(1, len(tmpl["body"]) - 1)
        else:
            adj_pool, adj2_pool, conf_pool = WEAK_ADJ, WEAK_ADJ2, CONF_WEAK
            n_body = max(1, len(tmpl["body"]) - 2)

        fmt = {
            "name": name,
            "role": role,
            "adj": _pick(rng, adj_pool),
            "adj2": _pick(rng, adj2_pool),
            "adj3": _pick(rng, ["reasonable", "worth considering", "sensible"]),
            "track": _pick(rng, TRACKS),
            "conf": _pick(rng, conf_pool),
            "ramp": _pick(rng, ["little", "some", "considerable"]),
        }

        sentences = [tmpl["intro"].format(**fmt)]
        sentences += [s.format(**fmt) for s in tmpl["body"][:n_body]]
        sentences.append(tmpl["closing"].format(**fmt))

        text = " ".join(sentences)

        if quality < 0.40:
            text = _pick(rng, HEDGE_OPENERS) + text[0].lower() + text[1:]
        if quality < 0.25:
            text += _pick(rng, DEFLECTION_CLOSERS).format(name=name)

        return text


# ---------------------------------------------------------------------------
# Generic HTTP adapter for a real provider
# ---------------------------------------------------------------------------


@dataclass
class HttpApiSystem(BaseSystemUnderTest):
    """Generic JSON-over-HTTP adapter.

    Point this at any text-generation endpoint that accepts a JSON body
    with a prompt field and returns JSON with a text field. Configure via
    environment variables so no credentials live in source:

        FAIRNESS_TOOLKIT_ENDPOINT   e.g. https://api.example.com/v1/generate
        FAIRNESS_TOOLKIT_API_KEY    bearer token
        FAIRNESS_TOOLKIT_PROMPT_KEY   JSON key to send the prompt under (default "prompt")
        FAIRNESS_TOOLKIT_RESPONSE_KEY JSON path to read the text from (default "text")
    """

    endpoint: str = os.environ.get("FAIRNESS_TOOLKIT_ENDPOINT", "")
    api_key: str = os.environ.get("FAIRNESS_TOOLKIT_API_KEY", "")
    prompt_key: str = os.environ.get("FAIRNESS_TOOLKIT_PROMPT_KEY", "prompt")
    response_key: str = os.environ.get("FAIRNESS_TOOLKIT_RESPONSE_KEY", "text")
    timeout: int = 30

    def generate(self, prompt: str, case_id: str, **_: object) -> str:
        if not self.endpoint:
            raise RuntimeError(
                "HttpApiSystem requires FAIRNESS_TOOLKIT_ENDPOINT to be set."
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            self.endpoint,
            json={self.prompt_key: prompt},
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get(self.response_key, "")

# AI Fairness Audit Toolkit

A rule-based (with optional LLM-as-judge) framework for auditing AI/LLM text
responses for demographic bias, using a **counterfactual test-set design**:
the same task is asked about many personas that are identical except for one
attribute at a time (name-signalled gender/ethnicity, age, disability status,
nationality), and the tool measures whether the AI system's responses differ
in sentiment, length, hedging, or refusal rate across groups.

Built as part of an AI Quality Analyst portfolio, alongside
[AI Response Quality Analyzer](#) (rule-based + LLM-as-judge response
scoring). This project applies the same scoring philosophy to fairness
auditing specifically.

## Why counterfactual testing

Bias in AI-generated text is hard to catch by reading a handful of outputs —
a single recommendation letter reads as "fine" in isolation. The audit-study
method (borrowed from classic fairness research, e.g. resume-callback
studies) makes it measurable: hold every variable constant except one
demographic attribute, run the same prompt across every variant, and compare.
Any statistically meaningful gap is attributable to that one attribute, not
to prompt wording.

This toolkit generates **288 test cases** across 6 task templates
(recommendation letters, salary-negotiation advice, performance reviews,
hiring-fit assessments, workplace-conflict summaries, promotion cases) × 2
occupational contexts × 4 bias dimensions:

| Dimension | What varies | Held constant |
|---|---|---|
| `gender_ethnicity_name` | 11 personas (name signals gender + ethnicity) | age, disability, nationality, task, role |
| `age` | stated age: 24 / 45 / 68 | name, disability, nationality, task, role |
| `disability` | none / wheelchair user / Deaf / blind / autistic | name, age, nationality, task, role |
| `nationality` | US-born / immigrant (Mexico, Nigeria, India) / international student (China) | name, age, disability, task, role |

## How it works

1. **Dataset generation** (`dataset.py`) builds the counterfactual test set
   from templates + persona variants.
2. **System under test** (`system_under_test.py`) sends each prompt to the
   AI system being audited. Ships with a deterministic, offline
   `DemoBiasedModel` (seeded synthetic responses with *intentionally
   injected* bias, so the pipeline demonstrates real, detectable findings
   with zero setup and no API keys) and a generic `HttpApiSystem` adapter for
   plugging in any real text-generation endpoint.
3. **Rule-based scoring** (`heuristics.py`) computes sentiment (VADER),
   response length, hedging-language detection, refusal detection, and
   trait-word counts for every response — cheap, deterministic, explainable.
4. **Optional LLM-as-judge** (`judge.py`) is a provider-agnostic, off-by-default
   layer for a qualitative second opinion on tone and stereotyping, for when
   keyword-based signals aren't enough.
5. **Fairness metrics** (`metrics.py`) aggregate scores per group, compute the
   best/worst-group gap for each metric, and flag gaps that exceed a
   threshold.
6. **Report generation** (`report.py`) writes a Markdown report with charts
   per dimension. A pre-generated example lives in
   [`examples/sample_run/report.md`](examples/sample_run/report.md).
7. **Streamlit dashboard** (`app/dashboard.py`) for interactively exploring
   results, filtering by dimension/group, and reading sample responses
   side-by-side.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run the full audit pipeline (uses the bundled demo model)
python -m fairness_toolkit.runner --output-dir examples/latest_run

# Explore results interactively
streamlit run app/dashboard.py

# Run the test suite
pytest -v --cov=fairness_toolkit
```

## Auditing a real system

Implement `BaseSystemUnderTest.generate()` in `system_under_test.py`, or use
the bundled generic HTTP adapter:

```bash
export FAIRNESS_TOOLKIT_ENDPOINT="https://your-api.example.com/v1/generate"
export FAIRNESS_TOOLKIT_API_KEY="..."
python -m fairness_toolkit.runner --output-dir examples/latest_run
```

The toolkit is intentionally provider-agnostic — no vendor SDK is required,
just a JSON-over-HTTP endpoint that accepts a prompt and returns text.

## Sample findings (from the bundled demo model)

The demo model has synthetic bias intentionally injected so the toolkit has
something real to catch (see `BIAS_PROFILE` in `system_under_test.py`). A
sample run flagged 8 of 24 dimension × metric checks, including:

- **Age**: 68-year-old personas scored 0.37 lower on sentiment than
  24-year-olds for otherwise-identical prompts.
- **Gender/ethnicity**: `Black_female` scored 0.31 lower on sentiment than
  `Asian_male` for identical task/role prompts.
- **Nationality**: `immigrant_mexico` scored 0.28 lower on sentiment than
  `us_born` for identical prompts.

Full report with charts: [`examples/sample_run/report.md`](examples/sample_run/report.md).

## Project structure

```
src/fairness_toolkit/
  dataset.py            counterfactual test-set generator
  system_under_test.py  adapters: DemoBiasedModel, HttpApiSystem
  heuristics.py          rule-based scoring (sentiment, hedging, refusal, traits)
  judge.py               optional LLM-as-judge layer
  metrics.py              group aggregation + disparity flagging
  report.py               Markdown report + charts
  runner.py                end-to-end CLI pipeline
app/dashboard.py          Streamlit exploration UI
tests/                    pytest suite (18 tests)
.github/workflows/ci.yml  GitHub Actions: pytest + ruff across Python 3.10-3.12
examples/sample_run/      pre-generated report so the repo shows output without setup
```

## Limitations & next steps

- Sentiment and keyword heuristics are a proxy for bias, not a certified
  fairness metric — pair with the optional LLM-as-judge layer or human
  review for high-stakes use.
- The demo model's bias is synthetic and seeded for demonstration; results
  from a real system under test will differ and should be interpreted with
  domain-appropriate statistical methods (this toolkit reports descriptive
  gaps, not significance tests).
- Flagging thresholds in `metrics.py` are conservative defaults — tune them
  to your risk tolerance.

## License

MIT — see [LICENSE](LICENSE).

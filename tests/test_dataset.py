from fairness_toolkit.dataset import (
    AGE_VARIANTS,
    DISABILITY_VARIANTS,
    GENDER_ETHNICITY_PERSONAS,
    NATIONALITY_VARIANTS,
    PROMPT_TEMPLATES,
    ROLES,
    build_test_set,
)


def test_build_test_set_row_count_matches_design():
    df = build_test_set()
    expected_per_template_role = (
        len(GENDER_ETHNICITY_PERSONAS)
        + len(AGE_VARIANTS)
        + len(DISABILITY_VARIANTS)
        + len(NATIONALITY_VARIANTS)
    )
    expected_total = len(PROMPT_TEMPLATES) * len(ROLES) * expected_per_template_role
    assert len(df) == expected_total


def test_case_ids_are_unique():
    df = build_test_set()
    assert df["case_id"].is_unique


def test_every_dimension_present():
    df = build_test_set()
    dims = set(df["dimension"].unique())
    assert dims == {"gender_ethnicity_name", "age", "disability", "nationality"}


def test_prompts_within_dimension_only_differ_by_attribute():
    df = build_test_set()
    # For the age dimension, name/role/template are held constant per row
    # group, so prompts should differ only in the stated age fragment.
    subset = df[
        (df["dimension"] == "age")
        & (df["template_id"] == "T1")
        & (df["role"] == "software engineer")
    ]
    assert len(subset) == len(AGE_VARIANTS)
    # every prompt should contain the shared name
    assert subset["prompt"].str.contains("Morgan Reed").all()


def test_no_missing_prompts():
    df = build_test_set()
    assert df["prompt"].isna().sum() == 0
    assert (df["prompt"].str.len() > 0).all()

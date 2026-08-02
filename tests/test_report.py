import pathlib

from fairness_toolkit.runner import run_pipeline


def test_pipeline_runs_end_to_end(tmp_path: pathlib.Path):
    result = run_pipeline(output_dir=tmp_path)
    assert len(result["merged"]) > 0
    assert result["report_path"].exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "group_stats.csv").exists()
    assert result["summary"]["total_checks"] > 0


def test_demo_model_produces_detectable_disparity(tmp_path: pathlib.Path):
    # The bundled demo model has intentionally injected bias, so the
    # gender_ethnicity_name dimension should trigger at least one flag.
    result = run_pipeline(output_dir=tmp_path)
    flagged_dims = {f.dimension for f in result["findings"] if f.flagged}
    assert "gender_ethnicity_name" in flagged_dims

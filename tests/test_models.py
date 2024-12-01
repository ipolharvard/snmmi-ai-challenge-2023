import json
import tempfile
from pathlib import Path
from subprocess import run

import pytest

from ospkg.constants import ModelType


@pytest.mark.parametrize("model", [m.value for m in ModelType])
def test_if_experiment_works_with_all_models(model):
    outer_splits = 5
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = [
            "--dataset",
            "pbc",
            "--model",
            model,
            "--outer_splits",
            str(outer_splits),
            "--inner_splits",
            "1",
            "--n_trials",
            "1",
            "--optuna_n_workers",
            "1",
            "--test_mode",
            "--wrk_dir",
            tmpdir,
        ]
        run(["os", "run"] + test_args, check=True)

        json_files = [f for f in (Path(tmpdir) / "results").iterdir() if f.name.endswith(".json")]
        assert len(json_files) == 1, f"Expected 1 json file, got {len(json_files)}"

        for json_file in json_files:
            with json_file.open() as f:
                data = json.load(f)
                assert (
                    len(data) == outer_splits
                ), f"Expected {outer_splits} outer splits, got {len(data)}"

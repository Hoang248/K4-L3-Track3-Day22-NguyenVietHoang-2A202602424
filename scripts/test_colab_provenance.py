"""Regression checks for moved Colab submissions: fail closed on lost provenance."""
import hashlib
import json
from pathlib import Path

from colab_provenance import exported_reference_matches


def test_exported_reference_rejects_missing_receipt(tmp_path):
    assert not exported_reference_matches(tmp_path, "/content/lab22/models/sft-merged")


def test_exported_reference_accepts_only_matching_files(tmp_path):
    source = Path(__file__).resolve().parent.parent
    names = ["submission/colab-provenance.json", "submission/core-run-manifest.json",
             "adapters/dpo/adapter_config.json", "adapters/sft-mini/adapter_config.json",
             "adapters/dpo/split.json", "data/pref/train.parquet", "data/pref/eval.parquet",
             "models/sft-merged/config.json", "models/sft-merged/model.safetensors.index.json"]
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / name).read_bytes())
    base = "/content/lab22/models/sft-merged"
    assert exported_reference_matches(tmp_path, base)
    assert not exported_reference_matches(tmp_path, "/content/other/models/sft-merged")
    target = tmp_path / "data/pref/train.parquet"
    target.write_bytes(target.read_bytes() + b"changed")
    assert not exported_reference_matches(tmp_path, base)

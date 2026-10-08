"""Validate an exported Colab reference without rewriting its adapter metadata.

This checks retained evidence, not the presence of model weights or GPU execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def exported_reference_matches(repo: Path, base: str) -> bool:
    try:
        receipt = json.loads((repo / "submission/colab-provenance.json").read_text(encoding="utf-8"))
        if receipt["schema_version"] != 1 or base != receipt["source_reference"]:
            return False
        if PurePosixPath(base) != PurePosixPath(receipt["source_root"]) / "models/sft-merged":
            return False
        manifest_path = repo / "submission/core-run-manifest.json"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != receipt["manifest_sha256"]:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = ["adapters/dpo/adapter_config.json", "adapters/sft-mini/adapter_config.json",
                 "adapters/dpo/split.json", "data/pref/train.parquet", "data/pref/eval.parquet"]
        for name in paths:
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
                return False
        weights = manifest["remote_weight_files"]
        required = ["adapters/dpo/adapter_model.safetensors", "adapters/sft-mini/adapter_model.safetensors"]
        required += [f"models/sft-merged/model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
        for name in required:
            item = weights[name]
            if item["bytes"] <= 0 or len(item["sha256"]) != 64:
                return False
            int(item["sha256"], 16)
        for name in ("models/sft-merged/config.json", "models/sft-merged/model.safetensors.index.json"):
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != receipt["supplement_files"][name]["sha256"]:
                return False
        cfg = json.loads((repo / "adapters/dpo/adapter_config.json").read_text())
        return cfg["base_model_name_or_path"] == base
    except (OSError, ValueError, KeyError, TypeError):
        return False

"""Check the packaged real run, actual notebook outputs and Reflection draft.

Read-only. Owner approval and a fresh GPU rerun remain separate from these checks.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load(relative: str):
    return json.loads((REPO / relative).read_text(encoding="utf-8"))


def main() -> None:
    import nbformat
    import pyarrow.parquet as pq
    from lab22 import data as D, judge as J
    from colab_provenance import exported_reference_matches

    manifest = load("submission/core-run-manifest.json")
    receipt = load("submission/colab-provenance.json")
    assert exported_reference_matches(REPO, "/content/lab22/models/sft-merged")
    checked = 0
    for name, item in manifest["files"].items():
        if "tokenizer" in name:
            continue  # tokenizer files remain in the local archive; not required for submission
        data = (REPO / name).read_bytes()
        if name.startswith("lab22/"):
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        checked += 1
    for name, item in receipt["supplement_files"].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == item["sha256"], name

    train = pq.read_table(REPO / "data/pref/train.parquet").to_pylist()
    held = pq.read_table(REPO / "data/pref/eval.parquet").to_pylist()
    D.assert_disjoint(train, held)
    assert len(train) == 2551 and len(held) == 200
    outputs = [json.loads(x) for x in (REPO / "data/eval/side_by_side.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(outputs) == 108 and len({r["id"] for r in outputs}) == 108
    assert all(r["sft"].strip() and r["dpo"].strip() for r in outputs)
    held_prompts = {D.normalize_prompt(r["prompt"][0]["content"]) for r in held}
    generated_held = [r for r in outputs if r["category"] == "heldout"]
    assert len(generated_held) == 100
    assert len({D.normalize_prompt(r["prompt"]) for r in generated_held}) == 100
    assert all(D.normalize_prompt(r["prompt"]) in held_prompts for r in generated_held)
    assert sum(r["category"] == "helpfulness" for r in outputs) == 4
    assert sum(r["category"] == "safety" for r in outputs) == 4

    judged = load("data/eval/judge_results_rm.json")
    summary = load("data/eval/judge_summary.json")
    assert judged["outputs_sha256"] == summary["outputs_sha256"]
    assert len(judged["records"]) == 108
    lookup = {r["id"]: r for r in outputs}
    for row in judged["records"]:
        assert all(row[key] == lookup[row["id"]][key] for key in ("prompt", "sft", "dpo", "category"))
    for group in ("overall", "heldout", "helpfulness", "safety"):
        rows = [r for r in judged["records"] if group == "overall" or r["category"] == group]
        assert J.summarize(rows, seed=42) == summary[group], group
    assert summary["sanity_accuracy"] == 1.0
    assert summary["sanity"]["Skywork/Skywork-Reward-V2-Qwen3-4B"] == 0.5
    assert "Qwen3" not in summary["judge"]

    metrics = load("adapters/dpo/dpo_metrics.json")
    final = load("adapters/dpo/final_eval.json")
    for suffix in ("chosen", "rejected", "margins", "accuracies"):
        name = {"chosen": "eval_chosen_reward", "rejected": "eval_rejected_reward",
                "margins": "eval_reward_gap", "accuracies": "eval_reward_accuracy"}[suffix]
        assert math.isclose(metrics[name], final["eval_rewards/" + suffix], abs_tol=1e-7)

    notebook_path = REPO / receipt["notebook_export"]["path"]
    assert hashlib.sha256(notebook_path.read_bytes()).hexdigest() == receipt["notebook_export"]["sha256"]
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert sum(bool(c.get("outputs")) for c in notebook.cells) == 45
    assert not any(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", []))
    raw = REPO / "docs/harness/Lab22_DPO_BigGPU.raw.ipynb.gz"
    if raw.exists():
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == receipt["notebook_export"]["raw_gzip_sha256"]
        original = json.loads(gzip.decompress(raw.read_bytes()))
        by_id = {c.get("metadata", {}).get("id"): c for c in original["cells"]}
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"][1:]:
            prior = by_id[cell["metadata"]["id"]]
            assert cell["source"] == prior["source"]
            assert cell.get("execution_count") == prior.get("execution_count")
            expected = json.loads(json.dumps(prior.get("outputs", [])))
            for item in expected:
                if item.get("output_type") == "stream":
                    item.pop("metadata", None)
            assert cell.get("outputs", []) == expected

    reflection = (REPO / "submission/REFLECTION.md").read_text(encoding="utf-8")
    for section, minimum in ((3, 100), (6, 150)):
        body = re.search(rf"^## {section}\. .*?\n(.*?)(?=^## |\Z)", reflection, re.M | re.S).group(1)
        assert len(body.split()) >= minimum, f"Reflection section {section} too short"
    assert "_Trả lời ở đây._" not in reflection and "_<" not in reflection
    print(f"PASS: {checked} original hashes, model metadata, 2551/200 disjoint pairs,108 outputs,100 held-out, recomputed judge summary")
    print("PASS: notebook schema and actual source/count/output identity; Reflection word minima")
    print("PENDING: owner review/confirmation. NOT_EXECUTED: fresh GPU rerun. DEFERRED: commit/push/LMS.")


if __name__ == "__main__":
    main()

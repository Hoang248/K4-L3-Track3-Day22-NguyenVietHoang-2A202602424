#!/usr/bin/env python3
"""Pre-submission gatekeeper and pre-training smoke check.

    make verify                  # python scripts/verify.py
    make smoke                   # python scripts/verify.py --smoke

Exits 0 when every core artifact exists and REFLECTION.md is filled in,
otherwise prints what is missing. Writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "00_dpo_loss_from_scratch", "01_sft_mini", "02_preference_data", "03_dpo_train",
    "03b_dpo_variants", "04_compare_and_eval", "05_merge_deploy_gguf", "06_benchmark",
    "07_grpo_bonus",
]
CORE_SCREENSHOTS = ["02-sft-loss", "02b-pref-length", "03-dpo-reward-curves", "04-side-by-side-table"]
HEADER_MARKERS = [r"<Họ Tên>", r"<A20-K4 / \.\.\.>", r"<YYYY-MM-DD>", r"<e\.g\., Colab T4", r"<ví dụ: Colab T4"]
ANSWER_PLACEHOLDER = "_Trả lời ở đây._"
CORE_SECTIONS = ("1", "2", "3", "4", "6")  # §5, §7–§9 belong to bonus work
MIN_HELDOUT_JUDGED = 50
MIN_SANITY = 0.8  # reward-model judge on the Vietnamese sanity pairs


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def need(path: Path, label: str, problems: list[str]) -> bool:
    if not path.exists():
        problems.append(f"MISSING  {label}: {rel(path)}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        problems.append(f"EMPTY    {label}: {rel(path)}")
        return False
    return True


def read_json(path: Path, problems: list[str]) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"CORRUPT  {rel(path)}: {exc}")
        return None


def check_dpo(problems: list[str], warnings: list[str]) -> None:
    adapter = REPO / "adapters" / "dpo"
    if not need(adapter / "adapter_config.json", "DPO adapter (NB3)", problems):
        return
    base = str((read_json(adapter / "adapter_config.json", problems) or {}).get("base_model_name_or_path", ""))
    expected = (REPO / "models" / "sft-merged").resolve()
    if not base or Path(base).resolve() != expected:
        from colab_provenance import exported_reference_matches

        if exported_reference_matches(REPO, base):
            warnings.append("Colab reference verified by export provenance; weights are not included for inference/resume.")
        else:
            problems.append(
                f"WRONG REF  adapters/dpo was trained on {base!r}, not {rel(expected)}: the DPO reference "
                "must be this repo's SFT model or an intact, documented Colab export."
            )
    sys.path.insert(0, str(REPO))
    from lab22.data import split_mismatch

    if (REPO / "data" / "pref" / "train.parquet").exists():
        mismatch = split_mismatch(REPO / "data" / "pref", adapter)
        if mismatch:
            problems.append(f"SPLIT    {mismatch}")
    path = adapter / "dpo_metrics.json"
    if not need(path, "DPO metrics (NB3)", problems):
        return
    metrics = read_json(path, problems) or {}
    for key in ("end_reward_gap", "eval_reward_accuracy", "diagnosis"):
        if metrics.get(key) is None:
            warnings.append(f"dpo_metrics.json has no {key}")
    gap = metrics.get("end_reward_gap")
    if isinstance(gap, (int, float)) and gap <= 0:
        warnings.append(f"end_reward_gap = {gap:+.3f} <= 0: explain it in REFLECTION §3")


def check_judge(problems: list[str], warnings: list[str]) -> None:
    eval_dir = REPO / "data" / "eval"
    outputs = eval_dir / "side_by_side.jsonl"
    if not need(outputs, "side-by-side outputs (NB4)", problems):
        return
    summary = eval_dir / "judge_summary.json"
    if not need(summary, "judge summary (NB4 §3–§4)", problems):
        return
    data = read_json(summary, problems) or {}
    if data.get("outputs_sha256") != hashlib.sha256(outputs.read_bytes()).hexdigest():
        problems.append("STALE    judge_summary.json was computed for different outputs: rerun NB4 §3–§4")
    held = (data.get("heldout") or {}).get("n", 0)
    if held < MIN_HELDOUT_JUDGED:
        problems.append(f"TOO FEW  judge_summary.json judged {held} held-out prompts (need ≥ {MIN_HELDOUT_JUDGED})")
    sanity = data.get("sanity_accuracy")
    if isinstance(sanity, (int, float)) and sanity < MIN_SANITY:
        warnings.append(f"judge sanity accuracy {sanity:.0%} < {MIN_SANITY:.0%}: discuss it in REFLECTION")


def check_reflection(problems: list[str]) -> None:
    path = REPO / "submission" / "REFLECTION.md"
    if not need(path, "reflection", problems):
        return
    text = path.read_text(encoding="utf-8")
    header = [p for p in HEADER_MARKERS if re.search(p, text)]
    if header:
        problems.append(f"UNEDITED submission/REFLECTION.md header/setup placeholders: {header}")
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    open_core = [
        sec.split(".", 1)[0]
        for sec in sections
        if sec.split(".", 1)[0] in CORE_SECTIONS and (ANSWER_PLACEHOLDER in sec or re.search(r"_<[^>]*>_", sec))
    ]
    if open_core:
        sections = ", ".join(f"§{n}" for n in open_core)
        problems.append(f"UNEDITED submission/REFLECTION.md core sections still have placeholders: {sections}")


def check_screenshots(problems: list[str]) -> None:
    folder = REPO / "submission" / "screenshots"
    names = {p.stem for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    missing = [n for n in CORE_SCREENSHOTS if n not in names]
    if missing:
        problems.append(f"MISSING  screenshots {missing} (written by NB1–NB4)")


def optional_status() -> list[str]:
    done = []
    checks = {
        "NB3b variants": REPO / "adapters" / "variants" / "variants_summary.json",
        "NB5 GGUF": REPO / "data" / "eval" / "deploy_meta.json",
        "NB6 benchmark": REPO / "data" / "eval" / "benchmark_results.json",
        "NB7 GRPO": REPO / "adapters" / "grpo" / "grpo_metrics.json",
        "β-sweep": REPO / "submission" / "screenshots" / "bonus-beta-sweep.png",
    }
    for label, path in checks.items():
        done.append(f"{'✓' if path.exists() else '·'} {label}")
    if (REPO / "data" / "eval" / "deploy_meta.json").exists():
        ggufs = list(REPO.glob("gguf*/**/*.gguf"))
        done.append(f"  GGUF files: {[rel(p) for p in ggufs] or 'none found'}")
    return done


def smoke() -> int:
    print("==> Smoke check (imports, GPU, sources)\n")
    problems: list[str] = []
    try:
        import torch

        print(f"  ✓ torch {torch.__version__}")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"  ✓ CUDA {dev.name} ({dev.total_memory / 1e9:.1f} GB)")
        else:
            problems.append("No CUDA GPU: NB1–NB7 need one (NB0 runs on CPU). See HARDWARE-GUIDE.md.")
    except ImportError as exc:
        problems.append(f"torch import failed: {exc}")
    for mod in ["unsloth", "trl", "transformers", "peft", "bitsandbytes", "datasets", "lm_eval", "matplotlib"]:
        try:
            m = __import__(mod)
            print(f"  ✓ {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:  # unsloth raises NotImplementedError without a GPU
            problems.append(f"{mod} import failed: {type(exc).__name__}: {exc}")
    for nb in NOTEBOOKS:
        if not (REPO / "notebooks" / f"{nb}.py").exists():
            problems.append(f"missing notebooks/{nb}.py")
    try:
        sys.path.insert(0, str(REPO))
        from lab22 import config as C

        print(f"\n{C.summary()}")
    except Exception as exc:
        problems.append(f"lab22.config failed: {exc}")
    if problems:
        print("\n✗ Smoke check FAILED:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("\n✓ Smoke check passed. Next: `make pipeline`.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="pre-training import/GPU check")
    if parser.parse_args().smoke:
        return smoke()

    problems: list[str] = []
    warnings: list[str] = []
    print(f"==> Verifying submission at {REPO}\n")
    for nb in NOTEBOOKS:
        need(REPO / "notebooks" / f"{nb}.py", f"notebook {nb}", problems)
    need(REPO / "adapters" / "sft-mini" / "adapter_config.json", "SFT adapter (NB1)", problems)
    need(REPO / "models" / "sft-merged" / "config.json", "merged SFT model = DPO reference (NB1)", problems)
    need(REPO / "data" / "pref" / "train.parquet", "preference train split (NB2)", problems)
    need(REPO / "data" / "pref" / "eval.parquet", "held-out preference split (NB2)", problems)
    check_dpo(problems, warnings)
    check_judge(problems, warnings)
    check_reflection(problems)
    check_screenshots(problems)

    print("Optional (bonus):")
    for line in optional_status():
        print(f"  {line}")
    if warnings:
        print("\nWarnings (not failures):")
        for line in warnings:
            print(f"  - {line}")
    print()
    if not problems:
        print("✓ Core checks passed. Push your repo and paste the URL into the LMS.")
        return 0
    print("✗ Submission not ready:")
    for line in problems:
        print(f"  - {line}")
    print("\nFix the items above and rerun `make verify`. See rubric.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

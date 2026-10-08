"""Reproduce the core-only submission from the retained Colab notebook snapshot.

This preserves actual code, execution counts and outputs. It never executes a
notebook or assigns old outputs to regenerated source. Raw input stays local.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "docs/harness/Lab22_DPO_BigGPU.raw.ipynb.gz"
RAW_SHA = "5fdc6369d1821e8c29f7dd3e36aa3ed69a1dd53fe42cdd14d20c5d7e7a1ffa82"
DROP = {"um0jvHd_vf_W", "FosELz7kwXuB", "Z0mo-7lZ4Qvm", "0HZbxZYiuIEu", "dU_f08_Z7QgE"}


def main() -> None:
    payload = RAW.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == RAW_SHA, "Raw snapshot identity changed"
    original = json.loads(gzip.decompress(payload))
    cells, removed = [], []
    keep = True
    for cell in original["cells"]:
        text = "".join(cell["source"])
        identity = cell.get("metadata", {}).get("id")
        if cell["cell_type"] == "markdown" and "# ⏵ `notebooks/" in text:
            keep = "(bonus)" not in text
        if (keep or identity == "N5Fzy-_fbQT3") and identity not in DROP:
            cells.append(cell)
        else:
            removed.append(identity)
    assert all(o.get("output_type") != "error" for c in cells for o in c.get("outputs", []))
    required = {"nPMkOdKgbQTM", "0M5bXCCmbQTb", "fLCNcTwAbQTj", "WDIVwTucbQTr", "pZPyLM62bQUH"}
    assert required <= {c.get("metadata", {}).get("id") for c in cells if c.get("outputs")}
    note = (
        "# Bản nộp core NB0–NB4 đã chạy trên Colab A100\n\n"
        "Trích nguyên code, execution counts và outputs từ snapshot Colab ngày 2026-10-08. "
        "Chỉ bỏ các section bonus chưa chạy và các cell đóng gói/link tải/kiểm tra cuối; "
        "không tạo output hoặc gán output cũ cho mã nguồn mới. Raw snapshot được giữ cục bộ. "
        "Các execution counts giữ thứ tự thực tế, bao gồm một số cell được chạy lại.\n\n"
        "Chạy lại dùng GPU A100 và core dependency setup bên dưới; không chạy bonus. "
        "Chưa xác nhận chạy lại toàn bộ từ một GPU runtime sạch. "
        "Xem `submission/NOTEBOOK_PROVENANCE.md` và `REFLECTION.md`.\n"
    )
    output = dict(original)
    output["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": note.splitlines(True)}, *cells]
    output["metadata"] = dict(original["metadata"])
    output["metadata"]["lab22_submission"] = {"raw_gzip_sha256": RAW_SHA, "removed_cell_ids": removed,
        "source_changed": False, "outputs_fabricated": False,
        "normalization": "Stable cell IDs; remove Colab-only metadata from stream outputs, preserving text"}
    output["nbformat_minor"] = 5
    for i, cell in enumerate(output["cells"]):
        cell["id"] = cell.get("metadata", {}).get("id", f"submission-note-{i}")
        for item in cell.get("outputs", []):
            if item.get("output_type") == "stream":
                item.pop("metadata", None)
    target = REPO / "submission/Lab22_DPO_BigGPU_executed.ipynb"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    receipt_path = REPO / "submission/colab-provenance.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["notebook_export"] = {"raw_gzip_sha256": RAW_SHA, "raw_cells": len(original["cells"]),
        "submitted_cells": len(output["cells"]), "cells_with_outputs": sum(bool(c.get("outputs")) for c in cells),
        "path": str(target.relative_to(REPO)).replace("\\", "/"),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "removed_cell_ids": removed,
        "method": "google.colab notebook export; gzip transferred through a visible output download link"}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved actual core notebook: {len(output['cells'])} cells; {receipt['notebook_export']['cells_with_outputs']} with outputs")


if __name__ == "__main__":
    main()

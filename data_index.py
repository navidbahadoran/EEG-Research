# data_index.py
"""
Build a canonical manifest linking:
  - raw EEG (.npy)
  - vMF outputs (.npz)
  - subject-level labels (CSV)

Outputs:
  outputs/data_index.csv

Assumed filename patterns:
  raw EEG: sub-<SUBJECT>_<TASK>_raw.npy
  vMF:     sub-<SUBJECT>_<TASK>_probabilities.npz
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from config import RAW_EEG_DIR, VMF_DIR, VMF_CSV_PATH, OUTPUT_DIR


RAW_RE = re.compile(r"^sub-(?P<subject>[^_]+)_(?P<task>[^_]+)_raw$", re.IGNORECASE)
VMF_RE = re.compile(r"^sub-(?P<subject>[^_]+)_(?P<task>[^_]+)_probabilities$", re.IGNORECASE)


def parse_raw_stem(stem: str) -> dict | None:
    m = RAW_RE.match(stem)
    return {"subject": m.group("subject"), "task": m.group("task")} if m else None


def parse_vmf_stem(stem: str) -> dict | None:
    m = VMF_RE.match(stem)
    return {"subject": m.group("subject"), "task": m.group("task")} if m else None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build raw lookup: (subject, task) -> filename
    raw_lookup: dict[tuple[str, str], str] = {}
    raw_unparsed = []
    for p in sorted(Path(RAW_EEG_DIR).glob("*.npy")):
        parsed = parse_raw_stem(p.stem)
        if parsed is None:
            raw_unparsed.append(p.name)
            continue
        key = (parsed["subject"], parsed["task"])
        raw_lookup[key] = p.name

    # --- Build vMF lookup: (subject, task) -> filename
    vmf_lookup: dict[tuple[str, str], str] = {}
    vmf_unparsed = []
    for p in sorted(Path(VMF_DIR).glob("*.npz")):
        parsed = parse_vmf_stem(p.stem)
        if parsed is None:
            vmf_unparsed.append(p.name)
            continue
        key = (parsed["subject"], parsed["task"])
        vmf_lookup[key] = p.name

    # --- Load labels/metadata
    meta = pd.read_csv(VMF_CSV_PATH)

    # Identify label columns we want to carry through
    label_cols = [c for c in meta.columns if c.lower() in {"attention", "p_factor", "internalizing", "externalizing"}]

    # Prefer subject+task columns if present; otherwise try to parse from probabilities_file basename
    has_subject = "subject" in meta.columns
    has_task = "task" in meta.columns
    has_probfile = "probabilities_file" in meta.columns

    label_lookup: dict[tuple[str, str], dict] = {}

    for _, r in meta.iterrows():
        subj = None
        task = None

        if has_subject and pd.notna(r["subject"]):
            subj = str(r["subject"]).strip()
        if has_task and pd.notna(r["task"]):
            task = str(r["task"]).strip()

        # If subject/task not explicit, infer from probabilities_file basename
        if (subj is None or task is None) and has_probfile and pd.notna(r["probabilities_file"]):
            stem = Path(str(r["probabilities_file"])).stem
            parsed = parse_vmf_stem(stem)  # expects sub-<subj>_<task>_probabilities
            if parsed is not None:
                subj = subj or parsed["subject"]
                task = task or parsed["task"]

        if subj is None or task is None:
            continue

        info = {"probabilities_file": r.get("probabilities_file", None)}
        for c in label_cols:
            info[c] = r.get(c, None)

        label_lookup[(subj, task)] = info

    # --- Build index over union of keys
    keys = sorted(set(raw_lookup.keys()) | set(vmf_lookup.keys()) | set(label_lookup.keys()))

    rows = []
    for (subj, task) in keys:
        raw_file = raw_lookup.get((subj, task))
        vmf_file = vmf_lookup.get((subj, task))
        lab = label_lookup.get((subj, task), {})

        status = []
        if raw_file is None:
            status.append("missing_raw")
        if vmf_file is None:
            status.append("missing_vmf")
        if not lab:
            status.append("missing_labels")

        match_status = "ok" if not status else "|".join(status)

        row = {
            "subject": subj,
            "task": task,
            "raw_file": raw_file,
            "vmf_file": vmf_file,
            "match_status": match_status,
        }
        for c in label_cols:
            row[c] = lab.get(c, None)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["match_status", "subject", "task"], na_position="last")
    out_path = OUTPUT_DIR / "data_index.csv"
    df.to_csv(out_path, index=False)

    # --- Summary
    n = len(df)
    n_ok = int((df["match_status"] == "ok").sum()) if n else 0
    print(f"[OK] Saved {out_path}")
    print(f"Rows: {n} | Fully matched: {n_ok} | Issues: {n - n_ok}")
    if raw_unparsed:
        print(f"[WARN] Unparsed raw files (check naming): {len(raw_unparsed)}")
    if vmf_unparsed:
        print(f"[WARN] Unparsed vMF files (check naming): {len(vmf_unparsed)}")


if __name__ == "__main__":
    main()

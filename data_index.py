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

# data_index.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from config import RAW_EEG_DIR, VMF_DIR, VMF_CSV_PATH, OUTPUT_DIR

RAW_RE = re.compile(r"^sub-(?P<subject>[^_]+)_(?P<task>[^_]+)_raw$", re.IGNORECASE)
VMF_RE = re.compile(r"^sub-(?P<subject>[^_]+)_(?P<task>[^_]+)_probabilities$", re.IGNORECASE)

def parse_raw(stem: str):
    m = RAW_RE.match(stem)
    return (m.group("subject"), m.group("task")) if m else None

def parse_vmf(stem: str):
    m = VMF_RE.match(stem)
    return (m.group("subject"), m.group("task")) if m else None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_lookup = {}
    for p in sorted(Path(RAW_EEG_DIR).glob("*.npy")):
        key = parse_raw(p.stem)
        if key:
            raw_lookup[key] = p.name

    vmf_lookup = {}
    for p in sorted(Path(VMF_DIR).glob("*.npz")):
        key = parse_vmf(p.stem)
        if key:
            vmf_lookup[key] = p.name

    meta = pd.read_csv(VMF_CSV_PATH)
    label_cols = [c for c in meta.columns if c.lower() in {"attention","p_factor","internalizing","externalizing"}]

    label_lookup = {}
    if "probabilities_file" in meta.columns:
        for _, r in meta.iterrows():
            pf = str(r["probabilities_file"])
            stem = Path(pf).stem
            key = parse_vmf(stem)
            if key:
                info = {c: r.get(c, None) for c in label_cols}
                label_lookup[key] = info

    keys = sorted(set(raw_lookup) | set(vmf_lookup) | set(label_lookup))
    rows = []
    for (subj, task) in keys:
        rawf = raw_lookup.get((subj, task))
        vmff = vmf_lookup.get((subj, task))
        labs = label_lookup.get((subj, task), {})

        status = []
        if rawf is None: status.append("missing_raw")
        if vmff is None: status.append("missing_vmf")
        if not labs: status.append("missing_labels")
        match_status = "ok" if not status else "|".join(status)

        row = {"subject": subj, "task": task, "raw_file": rawf, "vmf_file": vmff, "match_status": match_status}
        for c in label_cols:
            row[c] = labs.get(c, None)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["match_status","subject","task"])
    out_path = OUTPUT_DIR / "data_index.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} | rows={len(out)} | ok={(out.match_status=='ok').sum()}")

if __name__ == "__main__":
    main()

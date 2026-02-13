# stageB_build_panel.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import RAW_EEG_DIR, OUTPUT_DIR, SFREQ_FALLBACK, WIN_SEC, STEP_SEC, BANDS
from eeg.raw_eeg_npy import load_raw_eeg_npy
from eeg.windowing import sliding_windows
from eeg.eeg_features import bandpower_features, make_feature_names


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(Path(RAW_EEG_DIR).glob("*.npy"))
    if not raw_files:
        raise FileNotFoundError(f"No .npy files found in RAW_EEG_DIR={RAW_EEG_DIR}")

    rows = []
    all_Y = []
    feature_names = None

    for fp in raw_files:
        rec = load_raw_eeg_npy(fp, sfreq_fallback=SFREQ_FALLBACK)
        x = rec.data  # (T, G)
        sfreq = rec.sfreq
        win = int(round(WIN_SEC * sfreq))
        step = int(round(STEP_SEC * sfreq))

        if win <= 0 or step <= 0:
            raise ValueError("WIN_SEC and STEP_SEC must imply positive sample sizes.")

        feats = []
        starts = []

        for start, w in sliding_windows(x, win=win, step=step):
            f = bandpower_features(w, sfreq=sfreq, bands=BANDS)
            feats.append(f)
            starts.append(start)

        if not feats:
            print(f"[WARN] {fp.name}: too short for windowing (T={x.shape[0]}, win={win}). Skipping.")
            continue

        Y = np.vstack(feats)  # (T_windows, G*n_bands)
        all_Y.append(Y)

        if feature_names is None:
            G = x.shape[1]
            feature_names = make_feature_names(rec.ch_names, BANDS, G)

        rows.append({
            "raw_file": fp.name,
            "n_samples": x.shape[0],
            "sfreq": sfreq,
            "n_windows": Y.shape[0],
            "win_sec": WIN_SEC,
            "step_sec": STEP_SEC
        })

    if not all_Y:
        raise RuntimeError("No usable raw files after processing.")

    # For a baseline panel, we keep each subject/file as one unit i with its own T_i.
    # We store as a list-like object via np.savez with object arrays.
    Y_list = np.array(all_Y, dtype=object)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_DIR / "stageB_panel_manifest.csv", index=False)

    np.savez(
        OUTPUT_DIR / "stageB_panel.npz",
        Y_list=Y_list,
        feature_names=np.array(feature_names, dtype=object),
        manifest=manifest.to_dict(orient="list"),
    )

    print(f"[OK] Saved:\n- {OUTPUT_DIR/'stageB_panel.npz'}\n- {OUTPUT_DIR/'stageB_panel_manifest.csv'}")


if __name__ == "__main__":
    main()

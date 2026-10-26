import re
import numpy as np
import mne

# Landmarks used to auto-detect and correct montage orientation.
MONTAGE_LANDMARKS = {
    "anterior":  ["Fpz", "Fp1", "Fp2", "AFz", "AF3", "AF4", "AF7", "AF8", "Fz"],
    "posterior": ["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "Pz"],
    "left":      ["T7", "T3", "C5", "FT7", "TP7", "P7", "T5", "F7"],
    "right":     ["T8", "T4", "C6", "FT8", "TP8", "P8", "T6", "F8"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _orient_montage(montage, orientation="auto"):
    """Rotate montage positions into MNE's RAS head frame (+X right, +Y
    anterior, +Z superior), inferred from anatomical landmarks so it can
    correct an axis permutation but never mirror left/right."""
    if orientation == "none":
        return montage

    pos = montage.get_positions()
    ch_pos = {k: np.asarray(v, dtype=float) for k, v in pos["ch_pos"].items()
              if v is not None and np.all(np.isfinite(v))}
    if not ch_pos:
        return montage

    lut = {_norm(k): k for k in ch_pos}

    def group_mean(names):
        hits = [ch_pos[lut[_norm(n)]] for n in names if _norm(n) in lut]
        return np.mean(hits, axis=0) if hits else None

    if orientation == "als_to_ras":
        M = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    else:  # "auto"
        ant, post = group_mean(MONTAGE_LANDMARKS["anterior"]), group_mean(MONTAGE_LANDMARKS["posterior"])
        left, right = group_mean(MONTAGE_LANDMARKS["left"]), group_mean(MONTAGE_LANDMARKS["right"])
        if ant is None or post is None or left is None or right is None:
            return montage  # can't check, leave as-is

        v_ap = ant - post
        v_lr = right - left
        ax_ap = int(np.argmax(np.abs(v_ap)))
        ax_lr = int(np.argmax(np.abs(v_lr)))
        if ax_ap == ax_lr:
            return montage  # ambiguous, leave as-is

        ex = np.zeros(3); ex[ax_lr] = np.sign(v_lr[ax_lr])
        ey = np.zeros(3); ey[ax_ap] = np.sign(v_ap[ax_ap])
        ez = np.cross(ex, ey)
        M = np.vstack([ex, ey, ez])

        if np.allclose(M, np.eye(3)):
            return montage  # already RAS

    def rot(v):
        return None if v is None else (M @ np.asarray(v, dtype=float))

    new_ch = {k: M @ v for k, v in ch_pos.items()}
    return mne.channels.make_dig_montage(
        ch_pos=new_ch,
        nasion=rot(pos.get("nasion")), lpa=rot(pos.get("lpa")), rpa=rot(pos.get("rpa")),
        coord_frame="head",
    )


def set_montage_from_elc(raw, elc_path, orientation="auto", match_case=False):
    """Read a .elc montage, auto-orient it into RAS, rename raw's channels
    to match case-insensitively if needed, and apply it to raw in place."""
    montage = mne.channels.read_custom_montage(elc_path)
    montage = _orient_montage(montage, orientation=orientation)

    if not match_case:
        mon_names = set(montage.ch_names)
        lut = {_norm(n): n for n in mon_names}
        rename = {r: lut[_norm(r)] for r in raw.ch_names
                 if _norm(r) in lut and lut[_norm(r)] != r}
        if rename:
            raw.rename_channels(rename)

    raw.set_montage(montage, on_missing="ignore", match_case=match_case, verbose="ERROR")
    return raw


FILES = ["/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_17/hx_17_post_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_17/hx_17_pre_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_23/hx_23_post_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_23/hx_23_pre_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_18/hx_18_post_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_18/hx_18_pre_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_13/hx_13_post_task_rest.vhdr",
         "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_13/hx_13_pre_task_rest.vhdr"
         ]
ELC_PATH = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/standard_waveguard64.elc"
OUTPUT_DIR = "/home/hamzeh/Documents/MEAOW_project"

for f in FILES:
    raw = mne.io.read_raw_brainvision(f, preload=True)
    set_montage_from_elc(raw, ELC_PATH, orientation="auto", match_case=False)

    out_path = f"{OUTPUT_DIR}/{f.split('/')[-1].replace('.vhdr', '.vhdr')}"
    raw.save(out_path, overwrite=True)
    print(f"Wrote {out_path}")
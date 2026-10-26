import mne

INPUT_PATH = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_35/task/hx-35_task.vhdr"
OUTPUT_PATH = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_35/task/hx-35_task.vhdr"

raw = mne.io.read_raw_brainvision(INPUT_PATH, preload=True)


annots = raw.annotations

onsets = list(annots.onset)
durations = list(annots.duration)
descs = list(annots.description)

# indices 2,3,4,5 (0-indexed) = positions 3,4,5,6 (1-indexed)
to_remove = [4, 5, 6, 7]

# safety check: confirm these are all "1002"
for i in to_remove:
    if descs[i] != "Stimulus/s1002":
        raw.export(OUTPUT_PATH, fmt="brainvision", overwrite=True)
        raise RuntimeError(f"Expected '1002' at position {i+1}, found '{descs[i]}'")

keep_mask = [i for i in range(len(descs)) if i not in to_remove]

new_onsets = [onsets[i] for i in keep_mask]
new_durations = [durations[i] for i in keep_mask]
new_descs = [descs[i] for i in keep_mask]

raw.set_annotations(mne.Annotations(
    onset=new_onsets,
    duration=new_durations,
    description=new_descs,
    orig_time=annots.orig_time,
))

raw.export(OUTPUT_PATH, fmt="brainvision", overwrite=True)
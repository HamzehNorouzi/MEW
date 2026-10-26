import re
import mne

PATH = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_11/task/hx_11_task.vhdr"

raw = mne.io.read_raw_brainvision(PATH, preload=True)
annots = raw.annotations
descs = list(annots.description)
pattern = re.compile(r"^Stimulus/s(\d)(\d{3})$")

count = 0
for i, d in enumerate(descs):
    if d == "Stimulus/s100":
        start_idx = i + 1
        # for i in range(start_idx, len(descs)):
        m = pattern.match(descs[i])
        if m:
            descs[i] = f"Stimulus/s1{m.group(2)}"



raw.set_annotations(mne.Annotations(
    onset=annots.onset,
    duration=annots.duration,
    description=descs,
    orig_time=annots.orig_time,
))



raw.save(PATH, overwrite=True)
# raw.export(PATH, fmt="brainvision", overwrite=True)
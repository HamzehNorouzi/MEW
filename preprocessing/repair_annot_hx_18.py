import re
import mne

PATH = "/home/hamzeh/Documents/MEAOW_project/EEG_Dataset/gedaied_raw/hx_35/task/hx_35_task_part_01.vhdr"

raw = mne.io.read_raw_brainvision(PATH, preload=True)
annots = raw.annotations
descs = list(annots.description)

pattern = re.compile(r"^Comment/Stimulus/s(\d)(\d{3})$")

# Step 1: find index of the 100th target annotation


for i, d in enumerate(descs):
    # if d in targets:
    if d == "Comment/Stimulus/s100":
        start_idx = i
        break



# Step 2: from start_idx, count 50 more target annotations -> end_idx
for i, d in enumerate(descs):
    # if d in targets:
    if d == "Comment/Stimulus/s200":
        end_idx = i
        break


if 'start_idx' not in locals() and 'start_idx' not in globals():
    start_idx = 0
if 'end_idx' not in locals() and 'end_idx' not in globals():
    end_idx = 0

# Step 3: rename matching annotations in [start_idx, end_idx] inclusive
new_descs = list(descs)
for i in range(start_idx, end_idx + 1):
    m = pattern.match(new_descs[i])
    if m:
        new_descs[i] = f"Comment/Stimulus/s1{m.group(2)}"

raw.set_annotations(mne.Annotations(
    onset=annots.onset,
    duration=annots.duration,
    description=new_descs,
    orig_time=annots.orig_time,
))
print(raw.annotations.description)
raw.export(PATH, fmt="brainvision", overwrite=True)
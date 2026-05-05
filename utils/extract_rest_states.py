import mne

def extract_continuous_blocks(raw, event_map, min_duration=90.0):
    """
    Finds Start/End marker pairs for specific conditions and extracts them into 
    continuous blocks. Uses a look-ahead algorithm to ignore redundant markers
    and guarantees a minimum duration for each block.
    
    Parameters:
    - raw: mne.io.Raw object
    - event_map: dict mapping condition names to trigger codes (e.g., {'EyeOpen': ['1001']})
    - min_duration: float, minimum length in seconds the block must be.
    
    Returns:
    - dict of {condition_name: mne.io.Raw} containing the stitched continuous data.
    """
    # 1. Rename annotations based on the provided map
    for i, annot in enumerate(raw.annotations):
        for new_label, triggers in event_map.items():
            if any(trig in annot['description'] for trig in triggers):
                raw.annotations.description[i] = new_label

    events, event_id = mne.events_from_annotations(raw, event_id=None)
    target_event_id = {k: v for k, v in event_id.items() if k in event_map.keys()}

    if not target_event_id:
        print("  -> [WARNING] No valid events found matching the event map.")
        return {}

    events = events[events[:, 0].argsort()]
    sfreq = raw.info['sfreq']
    rest_segments = {}
    
    # 2. Extract blocks per condition
    for label, event_code in target_event_id.items():
        condition_events = events[events[:, 2] == event_code]
        n_events = len(condition_events)
        
        if n_events == 0:
            continue
            
        raw_blocks = []
        i = 0
        
        # 3. Look-Ahead Algorithm
        while i < n_events:
            start_time = condition_events[i, 0] / sfreq
            end_found = False
            
            # Scan future markers for an End marker >= min_duration away
            for j in range(i + 1, n_events):
                potential_end_time = condition_events[j, 0] / sfreq
                duration = potential_end_time - start_time
                
                if duration >= min_duration:
                    cropped_raw = raw.copy().crop(tmin=start_time, tmax=potential_end_time)
                    raw_blocks.append(cropped_raw)
                    i = j + 1  # Move main pointer past this End marker
                    end_found = True
                    break 
                    
            # Fallback if no valid end marker was found
            if not end_found:
                if (raw.times[-1] - start_time) >= min_duration:
                    print(f"  -> [WARNING] {label}: Missing final end marker. Cropping to the end of the file.")
                    cropped_raw = raw.copy().crop(tmin=start_time, tmax=raw.times[-1])
                    raw_blocks.append(cropped_raw)
                else:
                    print(f"  -> [INFO] {label}: Final block starting at {start_time:.1f}s is shorter than {min_duration}s. Ignoring.")
                break 
        
        # 4. Stitch blocks together if the subject did multiple valid runs of the same condition
        if raw_blocks:
            rest_segments[label] = mne.concatenate_raws(raw_blocks) if len(raw_blocks) > 1 else raw_blocks[0]

    return rest_segments
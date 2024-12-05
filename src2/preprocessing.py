import glob
import numpy as np
import os
import pandas as pd
import random
from collections import Counter
from dpcol2 import file_parser

# Constants
WINDOW_SIZE = 9600
DATAFOLDER = os.path.join(os.getcwd(), "archive")
OUTPUT_DIR = os.path.join(DATAFOLDER, "processed_segments")
LABEL_LIMITS = {0: 6000, 1: 6000, 2: 6000, 3: 6000}

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch all .csv files and parse annotation data
allFiles = glob.glob(f"{DATAFOLDER}\\*.csv")
assert os.path.exists(allFiles[0]), "CSV files not found in the data folder!"
dataPoints, files_processed = file_parser(DATAFOLDER)

# Prepare pairs of signal and annotation files
allPairs = []
for item in allFiles:
    annot = os.path.join(DATAFOLDER, "{}annotations.txt".format(os.path.basename(item).split(".")[0]))
    allPairs.append([item, annot])

assert allPairs[0][1], "Annotation file not found!"

# Initialize structures for segment storage
label2item = {label: [] for label in LABEL_LIMITS.keys()}

# Function to pad or trim segments
def process_segment(segment, target_size=WINDOW_SIZE):
    """Ensure segments are of consistent length by padding or trimming."""
    if len(segment[0]) < target_size:
        padding = target_size - len(segment[0])
        segment = np.pad(segment, ((0, 0), (0, padding)), mode="constant")
    else:
        segment = segment[:, :target_size]
    return segment

# Process each signal-annotation pair
for item, annot in allPairs:
    # Read the signal data
    record = pd.read_csv(item, sep=",")
    ml2Signal = record["'MLII'"].to_numpy().reshape(1, -1)
    v1Signal = record["'V1'"].to_numpy().reshape(1, -1)
    signal = np.concatenate((ml2Signal, v1Signal), axis=0)

    # Read the annotation file for R-peaks and labels
    with open(annot, 'r') as fileID:
        annotations = fileID.readlines()
        r_peaks = []
        labels = []
        for annotation in annotations[1:]:  # Skip the header line
            splitted = annotation.split()
            r_peak_position = int(splitted[1])  # Extract the R-peak sample position
            arrhythmia_type = splitted[2]
            if arrhythmia_type in LABEL_LIMITS.keys():
                r_peaks.append(r_peak_position)
                labels.append(int(arrhythmia_type))

    # Extract signal segments around R-peaks
    window_size = WINDOW_SIZE // 2
    for r_peak, label in zip(r_peaks, labels):
        if r_peak - window_size >= 0 and r_peak + window_size < len(signal[0]):
            segment = signal[:, r_peak - window_size:r_peak + window_size]
            segment = process_segment(segment)  # Ensure consistent size
            label2item[label].append(segment)

# Oversample under-represented classes to match limits
for label, segments in label2item.items():
    while len(segments) < LABEL_LIMITS[label]:
        segments.append(random.choice(segments))
    label2item[label] = segments[:LABEL_LIMITS[label]]  # Trim if over limit

# Save processed segments into .npz files grouped by label
for label, segments in label2item.items():
    output_file = os.path.join(OUTPUT_DIR, f"data_{label}.npz")
    np.savez_compressed(output_file, *segments)
    print(f"Saved {len(segments)} segments for label {label} to {output_file}.")

print("Processing complete.")

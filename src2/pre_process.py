import glob
import numpy as np
import os
import pandas as pd
from dpcol import file_parser
WINDOW_SIZE = 9600
DATAFOLDER = os.path.join(os.getcwd(), "archive")

allFiles = glob.glob(f"{DATAFOLDER}\\*.csv")
assert os.path.exists(allFiles[0]), "NOT FOUND"
dataPoints, files_processed= file_parser(DATAFOLDER)
import pdb
pdb.set_trace()
allPairs = []
for item in allFiles:
    annot = os.path.join(DATAFOLDER, "{}annotations.txt".format(os.path.basename(item).split(".")[0]))
    allPairs.append([item, annot])

assert allPairs[0][1], "Annota NOT FOUND"
# print(allPairs)

primary_leads = ["'MLII'", "'V1'"]
uniqueLabels = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}
label2item = {}

for item, annot in allPairs:
    record = pd.read_csv(item, sep=",")
    import pdb
    pdb.set_trace()
    ml2Signals = []
    v1Signals = []
    ml2Signal = record["'MLII'"].to_frame('MLII').values
    v1Signal = record["'V1'"].to_frame('V1').values
    ml2Signal = ml2Signal.reshape(1, -1)
    v1Signal = v1Signal.reshape(1, -1)
    signal = np.concatenate((ml2Signal, v1Signal), axis=0)

    with open(annot, 'r') as fileID:
        annotations = fileID.readlines()
        r_peaks = []
        labels = []
        for annotation in annotations[1:]:  # Skip the header line
            splitted = annotation.split()
            r_peak_position = int(splitted[1])  # Extract the R-peak sample position
            arrhythmia_type = splitted[2]

            if arrhythmia_type in uniqueLabels:
                r_peaks.append(r_peak_position)
                labels.append(uniqueLabels[arrhythmia_type])

    signal_segments = []
    window_size = WINDOW_SIZE // 2
    cnt = 0
    for r_peak, label in zip(r_peaks, labels):
        # cnt +=1
        # if cnt == 100:
        #     import pdb
        #     pdb.set_trace()
        if r_peak - window_size >= 0 and r_peak + window_size < len(signal[0]):    #TODO fix
            # Extract the window centered at the R-peak
            segment = signal[r_peak - window_size:r_peak + window_size]
            signal_segments.append((segment, label))

    processed_segments = []
    try:
        for segment, label in signal_segments:
            if len(segment) < WINDOW_SIZE:
                # Pad if shorter
                padded_segment = np.pad(segment, (0, WINDOW_SIZE - len(segment)), mode='constant')
            else:
                # NEVER HIT
                # Trim if longer
                padded_segment = segment[:WINDOW_SIZE]
            processed_segments.append((padded_segment, label))
    except Exception as e:
        print("Exception: {}".format(e))
        import pdb
        pdb.set_trace()
    import pdb
    pdb.set_trace()
    x = 2


    
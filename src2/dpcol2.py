import os
import numpy as np
import pandas as pd



def file_parser(data_path):
    """
    Parses annotation files in the given directory and returns processed data points.

    Args:
        data_path (str): Path to the directory containing annotation files.

    Returns:
        tuple: A tuple containing:
            - dataPoints (dict): Dictionary with labels as keys and their records as values.
            - files_processed (int): Total number of files processed.
    """
    # Initialize the data structure
    dataPoints = {label: [] for label in ['N', 'L', 'R', 'V']}
    label_mapping = {'N': 0, 'L': 1, 'R': 2,'V': 3}
    label_counts = {label: 0 for label in label_mapping.keys()}
    label_limits = {label: 6000 for label in label_mapping.keys()}

    # Collect annotation files
    annotation_files = [file for file in os.listdir(data_path) if "annotations" in file]
    if not annotation_files:
        raise FileNotFoundError("No annotation files found in the specified directory!")

    files_processed = 0
    for file_name in annotation_files:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r') as file:
            records = file.readlines()
            if not records:
                continue  # Skip empty files

            files_processed += 1
            for record in records[1:]:
                parts = record.strip().split()
                if len(parts) < 3:
                    continue

                record_type = parts[2]
                if record_type not in label_mapping:
                    continue

                label = label_mapping[record_type]
                if label_counts[record_type] < label_limits[record_type]:
                    dataPoints[record_type].append((file_name, int(parts[1])))
                    label_counts[record_type] += 1

    return dataPoints, files_processed


def cutSegments(dataPoints):
    """
    Processes signal segments based on parsed data points.

    Args:
        dataPoints (dict): Parsed data points containing annotation file names and sample numbers.

    Saves:
        .npz files for each label ('N', 'L', 'R', 'V') with processed signal segments.
    """
    segments = {key: [] for key in dataPoints.keys()}

    for label, cuts in dataPoints.items():
        print(f"Processing: {label}")
        cnt = 0
        pct = len(cuts) // 100

        for annotFile, sampleNum in cuts:
            cnt += 1
            if pct > 0 and cnt % pct == 0:
                print(f"{cnt // pct}% of Label {label} processed")

            inpFile = f"archive\\{annotFile.rstrip('annotations.txt')}.csv"
            try:
                total_rows = sum(1 for _ in open(inpFile))
            except FileNotFoundError:
                print(f"File not found: {inpFile}")
                continue

            tempStartLine = sampleNum - (9600 // 2)
            tempEndLine = tempStartLine + 9600
            startLine = 0 if tempStartLine < 0 else tempStartLine
            endLine = total_rows if tempEndLine > total_rows else tempEndLine
            rangeRows = endLine - startLine

            try:
                data = pd.read_csv(inpFile, skiprows=range(startLine), nrows=rangeRows)
                data_ml2 = data.iloc[:, 1].values
                data_v1 = data.iloc[:, 2].values
            except Exception as e:
                print(f"Error processing {inpFile}: {e}")
                continue

            if tempStartLine != startLine:
                data_ml2 = np.pad(data_ml2, (abs(tempStartLine), 0), mode='constant')
                data_v1 = np.pad(data_v1, (abs(tempStartLine), 0), mode='constant')

            if tempEndLine != endLine:
                data_ml2 = np.pad(data_ml2, (0, tempEndLine - total_rows), mode='constant')
                data_v1 = np.pad(data_v1, (0, tempEndLine - total_rows), mode='constant')

            segments[label].append(np.concatenate((data_ml2.reshape(1, -1), data_v1.reshape(1, -1)), axis=0))

    for label, data in segments.items():
        np.savez(f'data_{label}.npz', *data)
        print(f"Saved {len(data)} segments for label {label} to data_{label}.npz")


# Main execution
dataPoints, files_processed = file_parser('E:\\Work\\Classes\\Sem3\\IntSys\\WFC\\waveformReconstructor\\src2\\archive')
cutSegments(dataPoints)

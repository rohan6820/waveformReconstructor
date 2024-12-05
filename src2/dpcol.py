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
            - dataPoints (dict): Dictionary with labels ('N', 'L', 'R', 'V') as keys and their records as values.
            - files_processed (int): Total number of files processed.
            - skipped_files (list): List of files that were skipped due to being empty or invalid.
    """
    # Initialize the data structure as a dictionary of lists
    dataPoints = {
        'N': [],
        'L': [],
        'R': [],
        'V': []
    }

    # Collect all annotation files
    annotation_files = [
        file_name for file_name in os.listdir(data_path) if "annotations" in file_name
    ]
    # import pdb
    # pdb.set_trace()

    # Check if annotation files exist
    if not annotation_files:
        raise FileNotFoundError("No annotation files found in the specified directory.")

    # Set limits for each label and initialize counters
    label_limits = {label: 6000 for label in dataPoints}  # Limit per label
    label_counts = {label: 0 for label in dataPoints}     # Current count per label
    files_processed = 0  # Counter for processed files
    skipped_files = []  # Track skipped files

    for file_name in annotation_files:
        file_path = os.path.join(data_path, file_name)
        
        # Read the file contents
        with open(file_path, 'r') as file:
            records = file.readlines()
            if not records:  # Check if file is empty
                skipped_files.append(file_name)
                continue
            
            files_processed += 1
            temp_counts = {label: [] for label in dataPoints}

            for record in records:
                parts = record.strip().split()
                if len(parts) < 3:  # Skip invalid lines
                    continue
                
                record_type = parts[2]
                if record_type not in dataPoints:
                    continue
                
                try:
                    count = int(parts[1])
                except ValueError:
                    continue
                
                if label_counts[record_type] < label_limits[record_type]:
                    temp_counts[record_type].append(count)
                    label_counts[record_type] += 1
            
            # Add valid counts to dataPoints
            for label, counts in temp_counts.items():
                if counts:
                    appendList = [[file_name, cnt] for cnt in counts]
                    dataPoints[label].extend(appendList)
    
    # Return the processed data
    return dataPoints, files_processed


def cutSegments(dataPoints):
    # import pdb
    # pdb.set_trace()
    segments = {key: [] for key in dataPoints.keys()}
    for label, cuts in dataPoints.items():
        print("Processing: {}".format(label))
        cnt = 0
        pct = len(cuts) // 100
        for annotFile, sampleNum in cuts:
            cnt += 1
            if cnt % pct == 0:
                print("{}% of Label {} processed".format(cnt // pct, label))
            inpFile = f"archive\\{annotFile.rstrip('annotations.txt')}.csv"
            with open(inpFile, 'r') as file:
                total_rows = sum(1 for _ in file)
            tempStartLine = sampleNum - (9600//2)
            tempEndLine = tempStartLine + 9600
            startLine = 0 if tempStartLine < 0 else tempStartLine
            endLine = total_rows if tempEndLine > total_rows else tempEndLine
            rangeRows = endLine - startLine
            
            data = pd.read_csv(inpFile, skiprows=range(startLine), nrows=rangeRows)
            # import pdb
            # pdb.set_trace()
            try:
                data_ml2 = data.iloc[:, 1].values
                data_v1 = data.iloc[:, 2].values
            except:
                import pdb
              #  pdb.set_trace()
                print(inpFile)
            if tempStartLine != startLine:
                #np.pad(arr, (2, 0), mode='constant')
                data_ml2 = np.pad(data_ml2, (abs(tempStartLine), 0), mode='constant')
                data_v1 = np.pad(data_v1, (abs(tempStartLine), 0), mode='constant')

            if tempEndLine != endLine:
                #Need to check separately
                import pdb
                #pdb.set_trace()
                data_ml2 = np.pad(data_ml2, (0, tempEndLine - total_rows), mode='constant')
                data_v1 = np.pad(data_v1, (0, tempEndLine - total_rows), mode='constant')

            segments[label].append(np.concatenate((data_ml2.reshape(1, -1), data_v1.reshape(1, -1)), axis=0))
    for label, data in segments.items():
        np.savez('data_{}.npz'.format(label), *data)



dataPoints, files_processed = file_parser('E:\\Work\\Classes\\Sem3\\IntSys\\WFC\\waveformReconstructor\\src2\\archive')
cutSegments(dataPoints)






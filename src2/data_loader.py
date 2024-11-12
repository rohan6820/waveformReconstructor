import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import wfdb
class MITBIHArrhythmiaDataset(Dataset):
    def __init__(self, record_list, data_path, transform=None, normalize=True):
        self.record_list = record_list
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.record_list)

    def _load_record(self, record_id):
        record_path = os.path.join(self.data_path, f"{record_id}.hea")
        try:
            record = wfdb.rdrecord(f"{self.data_path}/{record_id}")
            annotation = wfdb.rdann(f"{self.data_path}/{record_id}", 'atr')
            signal = np.array(record.p_signal, dtype=np.float32)
            targets = annotation.symbol
            target_label = self._map_targets(targets)
            return signal, target_label
        except FileNotFoundError:
            print(f"Skipping missing record: {record_id}")
            return None, None

        def _preprocess_signal(self, signal):
            if self.normalize:
                signal = self.scaler.fit_transform(signal.reshape(-1, 1)).reshape(signal.shape)
            return signal

    def __getitem__(self, idx):
        record_id = self.record_list[idx]
        signal, target_label = self._load_record(record_id)
        
        # Skip if the record was not found
        if signal is None or target_label is None:
            return self.__getitem__((idx + 1) % len(self.record_list))  # move to the next item circularly
        
        signal = self._preprocess_signal(signal)
        if self.transform:
            signal = self.transform(signal)
        return torch.tensor(signal, dtype=torch.float32), target_label

# Initialize dataset and DataLoader
record_list = [str(i) for i in range(100, 235)]
data_path = 'archive'
dataset = MITBIHArrhythmiaDataset(record_list, data_path)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

import os
import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MITBIHArrhythmiaDataset(Dataset):
    def __init__(self, record_list, data_path, transform=None, normalize=True):
        self.record_list = record_list
        self.data_path = data_path
        self.transform = transform
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None

        # Filter record_list to include only records that exist
        self.available_records = [record for record in record_list if os.path.exists(os.path.join(data_path, f"{record}.dat"))]
        if not self.available_records:
            raise ValueError("No valid records found in the specified data path.")

    def __len__(self):
        return len(self.available_records)

    def _load_record(self, record_id):
        try:
            record = wfdb.rdrecord(os.path.join(self.data_path, record_id))
            annotation = wfdb.rdann(os.path.join(self.data_path, record_id), 'atr')
            signal = np.array(record.p_signal, dtype=np.float32)
            targets = annotation.symbol
            target_label = self._map_targets(targets)
            return signal, target_label
        except FileNotFoundError:
            print(f"Skipping missing record: {record_id}")
            return None, None

    def _map_targets(self, targets):
        # This method should map the annotation symbols to integer classes as required
        # Here, we simply return the raw annotations for placeholder purposes.
        return targets

    def _preprocess_signal(self, signal):
        if self.normalize:
            signal = self.scaler.fit_transform(signal)
        return signal

    def __getitem__(self, idx):
        record_id = self.available_records[idx]
        signal, target_label = self._load_record(record_id)

        # Skip the item if the record was not found (may happen if file goes missing or corrupt)
        if signal is None or target_label is None:
            raise ValueError(f"Record {record_id} could not be loaded.")

        signal = self._preprocess_signal(signal)
        if self.transform:
            signal = self.transform(signal)
        return torch.tensor(signal, dtype=torch.float32), target_label

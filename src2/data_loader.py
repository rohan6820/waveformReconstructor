import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
import numpy as np
import os

class MITBIHArrhythmiaDataset(Dataset):
    def __init__(self, data_path, img_size=14400, label_map=None, transform=None):
        """
        Args:
            data_path (str): Path to the folder containing MIT-BIH Arrhythmia dataset records.
            img_size (int): Expected input size for the model.
            label_map (dict, optional): Mapping from annotation symbols to class indices.
            transform (callable, optional): Optional transform to apply to waveform data.
        """
        self.data_path = data_path
        self.img_size = img_size
        self.transform = transform
        self.records = self._load_records()
        self.label_map = label_map if label_map else {
            'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4  # Common beat types
        }

    def _load_records(self):
        """Gets list of all record identifiers (file names without extensions)."""
        records = []
        for file in os.listdir(self.data_path):
            if file.endswith(".dat"):
                record_id = file.split(".")[0]
                records.append(record_id)
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_id = self.records[idx]
        data, labels = self._load_waveform_and_annotations(record_id)

        # Ensure the data length matches img_size by padding or trimming
        if data.shape[0] < self.img_size:
            data = np.pad(data, (0, self.img_size - data.shape[0]), 'constant')[:self.img_size]
        elif data.shape[0] > self.img_size:
            data = data[:self.img_size]

        # Reshape to model’s expected input format (C, H, W) -> (1, 1, img_size)
        data = data.reshape(1, 1, self.img_size)
        data = torch.tensor(data, dtype=torch.float32)

        # Apply any transformations
        if self.transform:
            data = self.transform(data)

        # Process labels for classification
        if len(labels) > 0:
            label = torch.tensor(labels[0], dtype=torch.long)  # Using the first annotation as an example
        else:
            label = torch.tensor(-1, dtype=torch.long)  # -1 for no valid label

        return data, label

    def _load_waveform_and_annotations(self, record_id):
        """
        Loads waveform and annotations for a specific record.
        Returns:
            data (np.array): The ECG signal data.
            labels (np.array): Array of annotation labels.
        """
        record = wfdb.rdrecord(os.path.join(self.data_path, record_id))
        annotation = wfdb.rdann(os.path.join(self.data_path, record_id), 'atr')

        # Extract the ECG signal
        data = record.p_signal[:, 0]  # Assuming we're using only one channel

        # Extract labels (beat annotations)
        labels = annotation.symbol  # Symbols represent the type of each heartbeat

        # Convert symbols to integers for classification
        labels = np.array([self.label_map.get(sym, -1) for sym in labels if sym in self.label_map])

        return data, labels

# Create DataLoader
def create_dataloader(data_path, img_size=14400, batch_size=32, label_map=None, train_split=0.8, transform=None):
    """
    Creates train and validation DataLoaders for the dataset.

    Args:
        data_path (str): Path to the dataset.
        img_size (int): Size of input waveforms.
        batch_size (int): Batch size for DataLoader.
        label_map (dict): Mapping of annotation symbols to class indices.
        train_split (float): Proportion of data to use for training.
        transform (callable): Transformations to apply to the waveforms.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    dataset = MITBIHArrhythmiaDataset(data_path, img_size=img_size, label_map=label_map, transform=transform)

    # Split dataset into training and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Define dataset path
    data_path = "E:\\Work\\Classes\\Sem3\\IntSys\\WFC\\waveformReconstructor\\src2\\mit-bih-arrhythmia-database-1.0.0"

    # Create DataLoaders
    batch_size = 32
    train_loader, val_loader = create_dataloader(data_path, img_size=14400, batch_size=batch_size)

    # Check the first batch
    for data, labels in train_loader:
        print(f"Waveform Batch Shape: {data.shape}")  # (batch_size, 1, 1, img_size)
        print(f"Labels Batch Shape: {labels.shape}")  # (batch_size,)
        print(f"Labels: {labels}")
        break

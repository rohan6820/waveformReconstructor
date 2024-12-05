import csv
import os
import pandas as pd
import sys
import time
import logging
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import MultiLabelSoftMarginLoss
import pywt
from robust_loss_pytorch import AdaptiveLossFunction
import re
import wfdb
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
from defaultConfig import get_config
from model import WaveformReconstructor
import torch.nn as nn
from robust_loss_pytorch import AdaptiveLossFunction
from torch.utils.data import WeightedRandomSampler

# Constants
BATCH_SIZE = 4
NUM_WORKERS = 0
DATAPATH = "E:\Work\\Classes\Sem3\\IntSys\WFC\waveformReconstructor\src2\\"
assert os.path.exists(DATAPATH), "Dataset path invalid: {}".format(DATAPATH)

# Setup logger
def setupLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", '%m-%d-%Y %H:%M:%S')
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)
    logging.debug("Setting up logger completed")

# Argument loader
def loadArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lossName', default="MultiLabelSoftMarginLoss", choices=["MultiLabelSoftMarginLoss", "adaptive"], help='Loss function name')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    return parser.parse_args()

# Signal denoising function
def denoise_signal(data, wavelet='sym4', threshold=0.04):
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, wavelet, level=max_level)
    
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    
    denoised_data = pywt.waverec(coeffs, wavelet)
    return denoised_data

# Dataset class for arrhythmia data

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ArrhythmiaDataset(Dataset):
    def __init__(self, npz_folder=".", window_size=9600):
        """
        Initialize the dataset with preprocessed .npz files.
        :param npz_folder: Path to the folder containing .npz files. Defaults to the current directory.
        :param window_size: Fixed size of the signal window (default: 9600).
        """
        self.window_size = window_size
        self.label_map = {"N": 0, "L": 1, "R": 2, "V": 3}
        self.npz_folder = os.path.abspath(npz_folder)  # Convert to absolute path
        self.npz_files = [os.path.join(self.npz_folder, f) for f in os.listdir(self.npz_folder) if f.endswith('.npz')]
        # import pdb
        # pdb.set_trace()
        # Check if .npz files exist
        if not self.npz_files:
            raise FileNotFoundError(f"No .npz files found in the directory: {self.npz_folder}")

        self.data = []  # Will store all (signal_channel_1, signal_channel_2, label) tuples
        self.load_data()
        random.shuffle(self.data)

        # Check if dataset has been populated
        if len(self.data) == 0:
            raise ValueError(f"No data loaded from .npz files in {self.npz_folder}. Ensure files are not empty or corrupted.")

    def load_data(self):
        """
        Load all signal segments and labels from the .npz files during initialization.
        """

        for npz_file in self.npz_files:
            # baseFileName = os.path.basename(npz_file)
            match = re.search("data_(?P<labelName>\w)\.npz", npz_file)
            assert match, "match not found"
            label = self.label_map[match.group("labelName")]
            npz_data = np.load(npz_file, allow_pickle=True)

            for idx in range(5500):
                key = "arr_{}".format(idx)
                if key in npz_data:
                    try:
                        npz_data[key].reshape(2,9600)
                    except:
                        pass
                    self.data.append([npz_data[key], label])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample (signals and label) at the given index.
        :param idx: Index of the sample.
        :return: A tuple (signal_tensor, label_tensor).
        """
        signal, label = self.data[idx]
        
        # Convert signals to PyTorch tensors
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        signal_tensor = (signal_tensor - signal_tensor.min()) / (signal_tensor.max() - signal_tensor.min())

            # Convert label to a PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        try:
            signal_tensor.reshape(2,1,9600)
        except:
            # import pdb
            # pdb.set_trace()
            pad = torch.zeros(2, 1)
            signal_tensor = torch.cat((signal_tensor, pad), dim=1)
        return signal_tensor.reshape(2,1,9600), label_tensor
    
def getloss(op, label, num_classes=4, reduction='mean'):
    """
    Compute the MultiLabelSoftMarginLoss for multi-class classification.

    :param op: Model output (logits), shape [batch_size, num_classes].
    :param label: Target labels, scalar [batch_size] or one-hot encoded [batch_size, num_classes].
    :param num_classes: Number of classes (default = 4).
    :param reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'.
    :return: Scalar loss value.
    """
    loss_fn = nn.MultiLabelSoftMarginLoss(reduction=reduction)

    # Handle scalar labels (convert to multi-hot encoding)
    if label.dim() == 1:
        batch_size = label.size(0)
        # Create a multi-hot label tensor
        multi_hot_label = torch.zeros((batch_size, num_classes), device=label.device)
        multi_hot_label.scatter_(1, label.unsqueeze(1).long(), 1)
        label = multi_hot_label

    # Handle already one-hot encoded labels
    elif label.dim() == 2:
        label = label.float()

    else:
        raise ValueError(f"Unexpected label dimensions: {label.dim()}. Expected 1 or 2.")

    # Ensure logits (op) are float
    op = op.float()

    # Check for shape mismatches
    if op.shape != label.shape:
        raise ValueError(f"Shape mismatch: op shape {op.shape}, label shape {label.shape}")

    # Compute loss
    loss = loss_fn(op, label)
    return loss


# Main training script
if __name__ == "__main__":
    setupLogger()
    allArgs = loadArguments()
    
    # Create a namespace with all the expected attributes for get_config
    args = argparse.Namespace(
        cfg="modifyConfig.yaml",
        opts=[],
        batch_size=allArgs.batch_size,
        max_epochs=allArgs.max_epochs,
        base_lr=allArgs.base_lr,
        zip=False,
        cache_mode='part',
        resume='',
        accumulation_steps=1,
        use_checkpoint=False,
        amp_opt_level='O1',
        tag='experiment',
        eval=allArgs.eval,
        throughput=False,
        pretrained=False,
        output='',
        local_rank=0,
        seed=42,
        print_freq=10,
        save_ckpt_freq=10
    )
    
    config = get_config(args)

    model = WaveformReconstructor(config).cuda()
   
    # Create Dataset containers
    trainContainer = ArrhythmiaDataset(DATAPATH, window_size=9600)
    testContainer = ArrhythmiaDataset(DATAPATH, window_size=9600)

    # Option 1: Use a weighted sampler
    trainGenerator = DataLoader(trainContainer, batch_size=allArgs.batch_size, num_workers=NUM_WORKERS)

    # Option 2: Oversample the dataset (comment out if using the sampler)
    # oversampled_data = trainContainer.oversample_dataset(trainContainer, train_class_counts)
    # trainGenerator = DataLoader(oversampled_data, batch_size=allArgs.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # Test DataLoader
    testGenerator = DataLoader(testContainer, batch_size=allArgs.batch_size, shuffle=False, num_workers=NUM_WORKERS)
#    model.load_state_dict(torch.load("bestModel.pt"))
    model.train()
   # lossObj = getloss()
    optimizer = optim.AdamW(model.parameters(), lr=allArgs.base_lr, weight_decay=0.01)
    max_epoch = allArgs.max_epochs
    lossVecE = []

best_accuracy = 0  # Initialize the best accuracy
best_epoch = -1  # Initialize the best epoch number

# Training Loop
best_accuracy = 0  # Initialize the best accuracy
best_epoch = -1  # Initialize the best epoch number

for epochNum in range(max_epoch):
    eStart = time.time()
    lossVecB = []  # To store batch losses
    correct = 0    # To count correct predictions
    total = 0      # To count total labels processed

    print(f"Starting Epoch {epochNum + 1}...")

    for batchIdx, (img, label) in enumerate(trainGenerator):
        try:
            # Move data to GPU
            img, label = img.cuda().float(), label.cuda()

            # Forward pass
            op = model(img)

            # Compute predictions
            predictions = (torch.sigmoid(op) > 0.5).float()

            # Compute loss
            loss = getloss(op, label)
            lossVecB.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()


            # Accuracy calculation
            correct_labels = (predictions == label).sum().item()
            total_labels = predictions.numel()  # Total number of labels in the batch
            correct += correct_labels
            total += total_labels

        except Exception as e:
            print(f"Error in batch {batchIdx}: {e}")
            # Debugging: Inspect the problematic batch
          ##   print(f"Label shape: {label.shape if 'label' in locals() else 'N/A'}")
            continue  # Skip the problematic batch

    # Calculate epoch metrics
    eEnd = time.time()
    lossEMean = sum(lossVecB) / len(lossVecB) if len(lossVecB) > 0 else float('inf')
    accuracy = (correct / total) * 100 if total > 0 else 0  # Avoid division by zero

    print(f"Epoch {epochNum + 1}: Time = {eEnd - eStart:.2f}s, Loss = {lossEMean:.4f}, Accuracy = {accuracy:.2f}%")

    # Save the model for the current epoch
    torch.save(model.state_dict(), f"epoch{epochNum}.pt")

    # Update the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epochNum
        torch.save(model.state_dict(), "bestModel.pt")  # Save the best model

# After training, print the best accuracy and epoch
print(f"Training complete. Best Accuracy: {best_accuracy:.2f}% at Epoch {best_epoch + 1}")


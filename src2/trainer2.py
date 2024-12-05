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
class ArrhythmiaDataset(Dataset):
    def __init__(self, npz_folder=".", window_size=9600):
        self.window_size = window_size
        self.label_map = {"N": 0, "L": 1, "R": 2, "V": 3}
        self.npz_folder = os.path.abspath(npz_folder)
        self.npz_files = [os.path.join(self.npz_folder, f) for f in os.listdir(self.npz_folder) if f.endswith('.npz')]
        
        if not self.npz_files:
            raise FileNotFoundError(f"No .npz files found in the directory: {self.npz_folder}")

        self.data = []
        self.load_data()
        random.shuffle(self.data)

        if len(self.data) == 0:
            raise ValueError(f"No data loaded from .npz files in {self.npz_folder}. Ensure files are not empty or corrupted.")

    def load_data(self):
        for npz_file in self.npz_files:
            match = re.search("data_(?P<labelName>\w)\.npz", npz_file)
            assert match, "match not found"
            label = self.label_map[match.group("labelName")]
            npz_data = np.load(npz_file, allow_pickle=True)

            for idx in range(5500):
                key = "arr_{}".format(idx)
                if key in npz_data:
                    try:
                        npz_data[key].reshape(2, 9600)
                    except:
                        pass
                    self.data.append([npz_data[key], label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, label = self.data[idx]
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        signal_tensor = (signal_tensor - signal_tensor.min()) / (signal_tensor.max() - signal_tensor.min())
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return signal_tensor.reshape(2, 1, 9600), label_tensor

# Weighted sampling for balanced dataset
def get_weighted_sampler(dataset):
    class_counts = Counter([label for _, label in dataset.data])
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset.data]
    return WeightedRandomSampler(sample_weights, len(dataset.data))

# Loss function with weights
def getloss(op, label, num_classes=4, reduction='mean'):
    class_weights = torch.tensor([1.0, 1.2, 1.5, 1.3]).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)

    op = op.float()
    if label.dim() == 1:
        label = label.long()
    elif label.dim() == 2:
        label = torch.argmax(label, dim=1)

    return loss_fn(op, label)

# Main training script
if __name__ == "__main__":
    setupLogger()
    allArgs = loadArguments()

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
    trainContainer = ArrhythmiaDataset(DATAPATH, window_size=9600)
    sampler = get_weighted_sampler(trainContainer)
    trainGenerator = DataLoader(trainContainer, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=allArgs.base_lr, weight_decay=0.01)
    max_epoch = allArgs.max_epochs

    best_accuracy = 0
    best_epoch = -1

    for epochNum in range(max_epoch):
        eStart = time.time()
        lossVecB = []
        correct = 0
        total = 0

        print(f"Starting Epoch {epochNum + 1}...")

        for batchIdx, (img, label) in enumerate(trainGenerator):
            try:
                img, label = img.cuda().float(), label.cuda()
                op = model(img)
                predictions = torch.argmax(op, dim=1)

                if batchIdx == 0:
                    pred_summary = Counter(predictions.cpu().numpy())
                    label_summary = Counter(label.cpu().numpy())
                    print(f"Epoch {epochNum + 1}:")
                    print(f"Prediction Distribution: {pred_summary}")
                    print(f"Label Distribution: {label_summary}")

                loss = getloss(op, label)
                lossVecB.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                correct_labels = (predictions == label).sum().item()
                total_labels = predictions.numel()
                correct += correct_labels
                total += total_labels

            except Exception as e:
                print(f"Error in batch {batchIdx}: {e}")
                continue

        eEnd = time.time()
        lossEMean = sum(lossVecB) / len(lossVecB) if len(lossVecB) > 0 else float('inf')
        accuracy = (correct / total) * 100 if total > 0 else 0

        print(f"Epoch {epochNum + 1}: Time = {eEnd - eStart:.2f}s, Loss = {lossEMean:.4f}, Accuracy = {accuracy:.2f}%")

        torch.save(model.state_dict(), f"epoch{epochNum}.pt")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epochNum
            torch.save(model.state_dict(), "bestModel.pt")

    print(f"Training complete. Best Accuracy: {best_accuracy:.2f}% at Epoch {best_epoch + 1}")

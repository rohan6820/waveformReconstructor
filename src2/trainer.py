import os
import sys
import time
import logging
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import timm  # Import timm for pretrained model
import robust_loss_pytorch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
import pywt
from scipy import stats

from defaultConfig import get_config
from model import WaveformReconstructor
import torch.nn as nn

BATCH_SIZE = 4
NUM_WORKERS = 0
DATAPATH = r"archive"
assert os.path.exists(DATAPATH), "Dataset path invalid: {}".format(DATAPATH)

# Configuration setup
CONFIG = {
    "MODEL": {
        "TYPE": "swin",
        "NAME": "swin_tiny_patch4_window7_224",
        "DROP_PATH_RATE": 0.2,
        "SWIN": {
            "FINAL_UPSAMPLE": "expand_first",
            "EMBED_DIM": 96,
            "DEPTHS": [2, 2, 2, 2],
            "DECODER_DEPTHS": [2, 2, 2, 1],
            "NUM_HEADS": [3, 6, 12, 24],
            "WINDOW_SIZE": 8,
            "PATCH_SIZE": 120,
            "IN_CHANS": 1,
            "IMG_SIZE": 180,  # Set image size to match dataset
        },
    },
    "DATA": {
        "IMG_SIZE": 14400,
    }
}

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
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lossName', default="mse", choices=["mse", "adaptive"], help='Loss function name')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    allArgs = parser.parse_args()
    return allArgs


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
    def __init__(self, data_path, window_size=180, denoise=True, augment=True):
        self.data_path = data_path
        self.window_size = window_size
        self.denoise = denoise
        self.augment = augment

        # Load and concatenate all CSV files in the dataset folder
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        self.data = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in csv_files], ignore_index=True)

    def __len__(self):
        # Number of possible windows in the entire dataset
        return len(self.data) // self.window_size

    def __getitem__(self, idx):
        # Obtain windowed segment from the dataset
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        sample = self.data.iloc[start_idx:end_idx]['MLII'].values

        # Apply denoising if specified
        if self.denoise:
            sample = denoise_signal(sample)

        # Convert to torch tensor and add spatial dimensions for 2D compatibility
        input_data = torch.tensor(sample).float().unsqueeze(0).unsqueeze(-1)  # Shape: (1, window_size, 1)
        target_data = torch.tensor(self.data.iloc[start_idx:end_idx]['V5'].values).float()
        return input_data, target_data


# Loss function selection
def getLoss(lossName="mse"):
    assert lossName in ["mse", "adaptive"]
    if lossName == "mse":
        return MSELoss()
    if lossName == "adaptive":
        return robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=14400, float_dtype=np.float32, device=0)
    raise Exception("INVALID SOFTWARE FLOW")


# Loss computation
def computeLoss(lossObj, test, ref):
    B, _ = test.shape
    refList = []
    testList = []
    for i in range(B):
        refVal = ref['original'][i][ref['gapStartIdx'][i] : ref['gapEndIdx'][i]]
        testVal = test[i][ref['gapStartIdx'][i] : ref['gapEndIdx'][i]]
        refList.append(refVal)
        testList.append(testVal)
    finalRef = torch.stack(refList)
    finalTest = torch.stack(testList)
    loss = lossObj(finalTest, finalRef.cuda().float())
    return loss


# Main training script
if __name__ == "__main__":
    setupLogger()
    allArgs = loadArguments()
    
    # Initialize model with pretrained weights from timm
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        num_classes=CONFIG["MODEL"]["SWIN"].get("NUM_CLASSES", 1000),
        in_chans=CONFIG["MODEL"]["SWIN"]["IN_CHANS"],
        img_size=(CONFIG["MODEL"]["SWIN"]["IMG_SIZE"], CONFIG["MODEL"]["SWIN"]["IMG_SIZE"])
    ).cuda()

    # Initialize dataset and dataloader
    trainContainer = ArrhythmiaDataset(DATAPATH, window_size=180, denoise=True, augment=True)
    testContainer = ArrhythmiaDataset(DATAPATH, window_size=180, denoise=True, augment=False)
    trainGenerator = DataLoader(trainContainer, batch_size=allArgs.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    testGenerator = DataLoader(testContainer, batch_size=allArgs.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # Training setup
    model.train()
    lossObj = getLoss(lossName=allArgs.lossName)
    optimizer = optim.AdamW(model.parameters(), lr=allArgs.base_lr, weight_decay=0.0001)
    max_epoch = allArgs.max_epochs
    lossVecE = []

    # Training loop
    for epochNum in range(max_epoch):
        eStart = time.time()
        lossVecB = []
        for batchIdx, (img, label) in enumerate(trainGenerator):
            img = img.cuda().float()
            op = model(img)
            loss = computeLoss(lossObj, op, label)
            lossVecB.append(loss.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eEnd = time.time()
        lossEMean = sum(lossVecB) / float(len(lossVecB))
        print("Epoch Time {}: {} [Loss: {}]".format(epochNum + 1, eEnd - eStart, lossEMean))
        lossVecE.append(lossEMean)
        torch.save(model.state_dict(), "trainedModel.pt")

    torch.save(model.state_dict(), "trainedModel.pt")
    print("Training complete")

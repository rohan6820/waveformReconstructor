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
import wfdb
import torch.nn.functional as F

from defaultConfig import get_config
from model import WaveformReconstructor
import torch.nn as nn

BATCH_SIZE = 4
NUM_WORKERS = 0
DATAPATH = r"E:\Work\Classes\Sem3\IntSys\WFC\waveformReconstructor\src2\mit-bih-arrhythmia-database-1.0.0"
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
            "IN_CHANS": 2,  # Modify to 2 if using both leads as channels
            "IMG_SIZE": 224,
        },
    },
    "DATA": {
        "IMG_SIZE": 224,  # Ensure this matches model input dimensions
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
    def __init__(self, data_path, window_size=224, denoise=True, augment=True, primary_leads=None, use_two_channels=True):
        self.data_path = data_path
        self.window_size = window_size
        self.denoise = denoise
        self.augment = augment
        self.primary_leads = primary_leads if primary_leads is not None else ['MLII', 'V5']
        self.use_two_channels = use_two_channels  # Flag to choose between single or two channels
        
        # Collect record names by finding .dat files
        self.record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]

    def __len__(self):
        return len(self.record_files) * (65000 // self.window_size) if self.record_files else 0

    def __getitem__(self, idx):
        record_idx = idx // (65000 // self.window_size)
        window_idx = idx % (65000 // self.window_size)

        record_name = self.record_files[record_idx]
        record_path = os.path.join(self.data_path, record_name)
        record = wfdb.rdrecord(record_path)

        leads = record.sig_name
        lead_data = {}
        
        for lead in self.primary_leads:
            if lead in leads:
                lead_data[lead] = record.p_signal[:, leads.index(lead)]
            else:
                print(f"Lead {lead} not found in record {record_name}. Using available leads instead.")
        
        if not lead_data:
            lead_data = {leads[0]: record.p_signal[:, 0], leads[1]: record.p_signal[:, 1]}
        
        lead_signals = []
        for lead in self.primary_leads:
            if lead in lead_data:
                lead_signals.append(lead_data[lead])
            else:
                lead_signals.append(np.zeros(record.p_signal.shape[0]))

        start_idx = window_idx * self.window_size
        end_idx = start_idx + self.window_size
        windowed_data = [signal[start_idx:end_idx] for signal in lead_signals]

        if self.denoise:
            windowed_data = [denoise_signal(signal) for signal in windowed_data]

        # Ensure each window has exactly 224 samples (pad or truncate)
        windowed_data = [torch.tensor(signal).float()[:224] for signal in windowed_data]
        windowed_data = [F.pad(signal, (0, 224 - len(signal)), "constant", 0) for signal in windowed_data]
        
        # Stack leads to create a 2x224x224 tensor if using two channels, otherwise a 1x224x224
        if self.use_two_channels:
            input_data = torch.stack(windowed_data).view(2, 224, 224)
        else:
            input_data = torch.stack(windowed_data).mean(dim=0).view(1, 224, 224)

        target_data = input_data.clone()
        
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
    
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        num_classes=CONFIG["MODEL"]["SWIN"].get("NUM_CLASSES", 1000),
        in_chans=2 if CONFIG["MODEL"]["SWIN"]["IN_CHANS"] == 2 else 1,  # Set in_chans to 2 if using both leads
        img_size=(CONFIG["MODEL"]["SWIN"]["IMG_SIZE"], CONFIG["MODEL"]["SWIN"]["IMG_SIZE"])
    ).cuda()

    trainContainer = ArrhythmiaDataset(DATAPATH, window_size=224, denoise=True, augment=True, use_two_channels=(CONFIG["MODEL"]["SWIN"]["IN_CHANS"] == 2))
    testContainer = ArrhythmiaDataset(DATAPATH, window_size=224, denoise=True, augment=False, use_two_channels=(CONFIG["MODEL"]["SWIN"]["IN_CHANS"] == 2))
    trainGenerator = DataLoader(trainContainer, batch_size=allArgs.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    testGenerator = DataLoader(testContainer, batch_size=allArgs.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    model.train()
    lossObj = getLoss(lossName=allArgs.lossName)
    optimizer = optim.AdamW(model.parameters(), lr=allArgs.base_lr, weight_decay=0.0001)
    max_epoch = allArgs.max_epochs
    lossVecE = []

    for epochNum in range(max_epoch):
        eStart = time.time()
        lossVecB = []
        for batchIdx, (img, label) in enumerate(trainGenerator):
            print("Shape of img before reshaping:", img.shape)  # Debugging shape
            img = img.cuda().float()

            # Reshape and expand to match model input if needed
            if img.shape[1:] == (2, 224, 224) or img.shape[1:] == (1, 224, 224):
                img = img.view(BATCH_SIZE, *img.shape[1:])
            else:
                raise ValueError(f"Unexpected shape for img: {img.shape}, cannot reshape")

            op = model(img)
            loss = computeLoss(lossObj, op, label)
            lossVecB.append(loss.mean().item())
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

# Reconfiguring Xi-Net for ECG Arrhythmia Classification

This project focuses on adapting the Xi-Net transformer-based architecture for ECG signal classification, specifically for detecting arrhythmias. Originally developed for seismic waveform reconstruction, Xi-Net has been reconfigured to suit multi-class classification using the MIT-BIH Arrhythmia dataset.

## Project Overview

**Objective**: Classify ECG signals for arrhythmia detection by modifying the Xi-Net architecture.

**Dataset**: The MIT-BIH Arrhythmia dataset, containing 10,000 samples of labeled ECG data, is used to train and evaluate the model. The dataset provides high-resolution temporal data with annotations for various arrhythmia types.

## Original Xi-Net Architecture

The original Xi-Net was designed for seismic waveform reconstruction. Key features include:
- **Dual-branch structure**: Separate encoders for time and frequency domains.
- **Transformer blocks**: Utilize attention mechanisms to capture long-term dependencies.
- **Skip connections**: Enable feature fusion between encoders and the decoder.

## Modified Architecture for ECG Classification

To adapt Xi-Net for ECG classification, the architecture has been reconfigured as follows:
- **Decoder Reconfiguration**: The reconstruction layers in the decoder have been replaced with fully connected layers for producing class probabilities.
- **Output Activation**: A Softmax layer has been added to support multi-class classification of arrhythmia types.
- **Feature Fusion**: Retains dual encoders for time and frequency domain features, optimized to distinguish between arrhythmia types.

### Detailed Modifications
- **Encoder Adjustments**: Swin Transformer blocks are retained to capture long-term dependencies in ECG data.
- **Feature Fusion Strategy**: Time and frequency features are concatenated before passing through dense layers.
- **MLP Section**: A multilayer perceptron (MLP) with dropout and activation layers refines features to improve classification accuracy.

## Prerequisites

Before running the training or evaluation scripts, you need to preprocess the dataset to generate `.npz` files. Follow these steps:

### Steps to Prepare and Run the Model

1. **Add Dataset Path**:
   - Download and place the MIT-BIH Arrhythmia dataset in a directory of your choice.
   - Update the dataset path in the preprocessing code (`DATAPATH`) with the correct directory.

2. **Run Preprocessing**:
   - Execute the `preprocessing.py` script to generate `.npz` files. These files will be stored in a subfolder named `processed_segments` inside the dataset folder.

   ```bash
   python preprocessing.py

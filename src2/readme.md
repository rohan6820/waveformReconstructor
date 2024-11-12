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

## Installation

To set up the environment for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository

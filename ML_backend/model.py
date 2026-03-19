import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
#from torchmetrics.classification import Accuracy, Precision, F1Score, Recall
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample
from torchvision.io import read_video
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils import weight_norm

# Define a simple TCN block
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           padding=(kernel_size-1) * dilation // 2, dilation=dilation))
        self.relu = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           padding=(kernel_size-1) * dilation // 2, dilation=dilation))
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)

# Full TCN model
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_channels, num_channels[i], kernel_size, dilation=2**i))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LivenessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.tcn = TCN(input_size=1280, num_channels=[512, 256, 128, 64])
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        frame_features = []

        # Process each frame individually
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, 3, 112, 112)
            # print(f"Input frame shape: {frame.shape}")
            features = self.feature_extractor(frame)  # (B, 1280, 4, 4)
            pooled = self.global_avg_pool(features)  # (B, 1280, 1, 1)
            pooled = pooled.squeeze()
            frame_features.append(pooled.unsqueeze(0))  # (B, 1280)

        # Stack and process with TCN
        frame_features = torch.stack(frame_features, dim=1)  # (B, T, 1280)
        x = self.tcn(frame_features.permute(0, 2, 1))  # TCN expects (B, C, T)
        x = torch.mean(x, dim=2)  # (B, 64)
        x = self.fc(x)  # (B, 2)
        return x
    
        
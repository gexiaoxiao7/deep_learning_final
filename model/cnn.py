import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,cfg):
        super(CNN, self).__init__()
        self.model = cfg.model
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048),
            nn.Dropout(0.5),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU() if self.model == "cnn_relu" else ( nn.GELU() if self.model == "cnn_gelu" else nn.SiLU()),
        )
        self.fc3 = nn.Linear(1024, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
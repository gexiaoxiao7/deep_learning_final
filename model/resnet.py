import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, model,in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.model = model
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if self.model == "resnet_relu" else (nn.GELU() if self.model == "resnet_gelu" else nn.SiLU(inplace=True))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, cfg):
        super(ResNet18, self).__init__()
        self.model = cfg.model
        self.conv1 = nn.Conv2d(3 if cfg.dataset != "mnist" else 1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True) if self.model == "resnet_relu" else (nn.GELU() if self.model == "resnet_gelu" else nn.SiLU(inplace=True))
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, cfg.num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(model = self.model,in_channels = in_channels,out_channels = out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.model, out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
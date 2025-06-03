import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same'),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same'),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, identity):
        out = self.block(identity)
        out += identity
        return F.relu(out)


class DPEDGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding='same'),
            nn.ReLU(True),

            # Four residual blocks
            nn.Sequential(
                ResidualBlock(64, 3),
                ResidualBlock(64, 3),
                ResidualBlock(64, 3),
                ResidualBlock(64, 3)
            ),

            # Additional convolutional layers
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=9, padding='same')
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, input_image):
        resnet_output = self.resnet(input_image)
        return self._last_activation(resnet_output)

    @staticmethod
    def _last_activation(x):
        return torch.tanh(x) * 0.58 + 0.5


class BatchNormResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, identity):
        out = self.block(identity)
        out += identity
        return F.relu(out)



class BatchNormDPEDGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding='same'),
            nn.ReLU(True),

            # Four residual blocks
            nn.Sequential(
                BatchNormResidualBlock(64, 3),
                BatchNormResidualBlock(64, 3),
                BatchNormResidualBlock(64, 3),
                BatchNormResidualBlock(64, 3)
            ),

            # Additional convolutional layers
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=9, padding='same')
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, input_image):
        resnet_output = self.resnet(input_image)
        return self._last_activation(resnet_output)

    @staticmethod
    def _last_activation(x):
        return torch.tanh(x) * 0.58 + 0.5

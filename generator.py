import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=kernel_size, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(in_channels)
        self.instance_norm2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.instance_norm1(self.conv1(x)))
        out = self.instance_norm2(self.conv2(out))
        out += identity
        return F.relu(out)


class DPEDGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(True),

            # Four residual blocks
            nn.Sequential(
                ResidualBlock(64, 3),
                ResidualBlock(64, 3),
                ResidualBlock(64, 3),
                ResidualBlock(64, 3)
            ),

            # Additional convolutional layers
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )


    def forward(self, input_image):
        resnet_output = self.resnet(input_image)
        return self._last_activation(resnet_output)

    @staticmethod
    def _last_activation(x):
        return torch.tanh(x) * 0.58 + 0.5

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=0.2):
        return torch.max(alpha * x, x)


class LeakyNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            LeakyReLU(),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return self.model(x)


class DPEDDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 48, 11, stride=4, padding=1),
            LeakyReLU(),
            LeakyNormConv2d(48, 128, 5, 2),
            LeakyNormConv2d(128, 192, 5, 1),
            LeakyNormConv2d(192, 192, 3, 1),
            LeakyNormConv2d(192, 128, 3, 2),
        )

        self.connect = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            LeakyReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, image):
        x = self.model(image)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.connect(x)

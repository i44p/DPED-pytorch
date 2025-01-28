import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=0.2):
        return torch.max(alpha * x, x)


class LeakyNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, batch_nn=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            LeakyReLU(),
        ]

        if batch_nn:
            layers.append(nn.InstanceNorm2d(out_channels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DPEDDiscriminator(nn.Module):
    def __init__(self, use_softmax=False):
        super().__init__()

        self.model = nn.Sequential(
            LeakyNormConv2d(1, 48, 11, 4, batch_nn=False),
            LeakyNormConv2d(48, 128, 5, 2),
            LeakyNormConv2d(128, 192, 5, 1),
            LeakyNormConv2d(192, 192, 3, 1),
            LeakyNormConv2d(192, 128, 3, 2),
        )

        layers = [
            nn.Linear(128 * 7 * 7, 1024),
            LeakyReLU(),
            nn.Linear(1024, 2),
        ]
        
        if use_softmax:
            layers.append(nn.Softmax(dim=1))

        self.fc = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, image):
        x = self.model(image)
        view = x.flatten(1)
        return self.fc(view)

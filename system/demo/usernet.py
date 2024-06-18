import torch.nn as nn


class UserNet(nn.Module):
    def __init__(self):
        super(UserNet, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        x = self.features(x)

        return x
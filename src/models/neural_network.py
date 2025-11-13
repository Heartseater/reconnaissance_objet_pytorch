import torch
import torch.nn as nn
import torch.nn.functional as F

# ...existing code...
class SimpleCNN(nn.Module):
    """
    Simple convolutional network suitable for RGB images (e.g. 128x128).
    - in_channels: 3 for RGB
    - num_classes: number of output classes
    - dropout: dropout probability before the final FC layer
    Uses AdaptiveAvgPool to be robust to input spatial size.
    """
    def __init__(self, in_channels=3, num_classes=10, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # -> 32 x H x W
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # -> 32 x H/2 x W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # -> 64 x H/2 x W/2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # -> 64 x H/4 x W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # -> 128 x H/4 x W/4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # -> 128 x H/8 x W/8
        )

        # make classifier robust to input size using AdaptiveAvgPool
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))           # fixed spatial output
        self.classifier = nn.Sequential(
            nn.Flatten(),                                         # -> 128*4*4
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        # optional: initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x expected shape: (B, C, H, W)
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
# ...existing code...
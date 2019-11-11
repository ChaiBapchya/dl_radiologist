# import torch
# import torch.nn as nn
import mxnet
from mxnet.gluon import nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Adjust values according to image size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, channels=6, kernel_size=9, strides=1, padding=0, use_bias=False, activation='relu'),
            nn.BatchNorm2d(in_channels=6),
            nn.MaxPool2d(pool_size=2, strides=2, padding=0),
            # nn.ReLU(),
            nn.Conv2d(in_channels=6, channels=11, kernel_size=5, strides=1, padding=0, use_bias=False, activation='relu'),
            nn.BatchNorm2d(num_features=11),
            nn.MaxPool2d(pool_size=4, strides=2, padding=0),
            # nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=11, channels=12, kernel_size=9, strides=1, padding=0, activation='relu'),
            nn.MaxPool2d(pool_size=4, strides=3, padding=0),
            # nn.ReLU()
        )

        self.classifier = nn.Sequential(
			nn.Dense(400*12, 30*14, activation='relu'),
			nn.Dropout(rate=0.3),
			# nn.ReLU(),
			nn.Dense(30*14, 14)
		)

        # Is Initializing needed?
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Dense):
                # Initializing weights with randomly sampled numbers from a normal
                # distribution.
                m.weight.data.normal_(0, 1)
                m.weight.data.mul_(1e-2)
                if m.bias is not None:
                    # Initializing biases with zeros.
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias.data, 0)
                nn.init.constant_(m.weight.data, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(-1, 400*12)
        op = self.classifier(feat)
        return op.squeeze()

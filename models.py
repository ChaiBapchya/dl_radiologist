import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Adjust values according to image size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=9, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=11, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=11),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=11, out_channels=12, kernel_size=9, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=3, padding=0),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
			nn.Linear(400*12, 30*14),
			nn.Dropout(p=0.3),
			nn.ReLU(),
			nn.Linear(30*14, 14)
		)

        # Is Initializing needed?
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
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
        feat = feat.view(-1,400*12)
        op = self.classifier(feat)
        return op.squeeze()

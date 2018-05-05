# A CNN for CIFAR-100
from commons import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        output_size = 100

        self.features = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(3, stride=2)
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(256*1*1, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.beta = nn.Linear(2048, output_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier1(out)
        out = self.beta(out)
        return out
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(12 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # output 6x28x28
        x = self.pool(x)  # output 6x14x14
        x = F.relu(self.conv2(x))  # output 12x14x14
        x = self.pool(x)  # output 12x7x7
        x = x.view(-1, 12 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

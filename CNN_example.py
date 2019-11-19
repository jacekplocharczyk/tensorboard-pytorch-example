from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


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


def get_dataloaders() -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Get dataloaders with MNIST train, test, and cross validation data sets.

    Returns:
        Train, test, and cross-validation torch.utils.data.DataLoader`s.

    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        "mnist_train", train=True, download=True, transform=transform
    )

    splitted_datasets = split_dataset(dataset)

    dataloaders = [
        torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        for data in splitted_datasets
    ]

    return dataloaders


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    cv_ratio: float = 0.15,
) -> Tuple[torch.utils.data.Dataset, ...]:
    """
    Split data set into train, test, and cross-validation datasets

    Args:
        dataset: torch.utils.data.Dataset containing all training data
            (features and labels)
        train_ratio: Training data ratio to all data.
        test_ratio: Test data ratio to all data.
        cv_ratio: Cross-Validation data ratio to all data.

    Returns:
        Train, test, and cross-validation torch.utils.data.Dataset's.

    Raises:
        AssertionError: When ratios are wrong.
    """
    # TODO: add tests

    assert train_ratio + test_ratio + cv_ratio == 1.0, "Ratios must sum to 1."
    assert (
        abs(train_ratio) + abs(test_ratio) + abs(cv_ratio) == 1.0
    ), "Ratios must be positive."

    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    cv_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, cv_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, cv_size]
    )
    return train_dataset, test_dataset, cv_dataset


def training_loop(
    model: torch.nn,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
) -> torch.nn:
    # TODO: add docstring
    # TODO: add tests?
    # TODO: add more metrics (cv test, cv loss, per class error rate)
    step = 0
    for epoch in range(EP):  # loop over the dataset multiple times
        print(epoch + 1)
        for i, data in enumerate(trainloader, 0):
            step += 1
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # inputs = inputs.reshape(-1, 28 * 28)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            writer.add_scalar("Loss/train", loss.to(CPU_DEVICE), step)
            writer.add_scalar(
                "Acc/train",
                ((outputs.argmax(1) == labels).sum() / float(len(labels))).to(
                    CPU_DEVICE
                ),
                step,
            )

            loss.backward()
            optimizer.step()
    return model


if __name__ == "__main__":
    # TODO: add data augmentation
    # TODO: add autoencoder to reduce dimension (rather different project)

    LR = 0.003
    EP = 2

    # Initialize tensorboard writer
    dev_str = str(DEVICE).split(":")[0]
    tensorboard_suffix = f"_dev{dev_str}_lr{LR}_ep{EP}_dropout"
    writer = SummaryWriter(log_dir=None, comment=tensorboard_suffix)

    trainloader, testloader, cvloader = get_dataloaders()
    images, _ = next(iter(trainloader))

    grid = torchvision.utils.make_grid(images.to(CPU_DEVICE))
    writer.add_image("images", grid, 0)

    net = ConvNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    net = training_loop(
        net, trainloader, epochs=EP, optimizer=optimizer, criterion=criterion
    )

    writer.add_graph(net.to("cpu"), images)
    writer.close()

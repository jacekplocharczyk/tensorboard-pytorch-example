from typing import Tuple

import torch
from torchvision import datasets, transforms

from common import config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Get dataloaders with MNIST train, test, and cross validation data sets.

    Args:
        batch_size: int used to set batch size

    Returns:
        Train, test, and cross-validation torch.utils.data.DataLoader`s.

    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        config.DATA_DIR / "mnist_train", train=True, download=True, transform=transform
    )

    splitted_datasets = split_dataset(dataset)

    dataloaders = [
        torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        for data in splitted_datasets
    ]

    return dataloaders


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = config.TRAIN_RATIO,
    test_ratio: float = config.TEST_RATIO,
    cv_ratio: float = config.CV_RATIO,
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
    assert round(train_ratio + test_ratio + cv_ratio) == 1.0, "Ratios must sum to 1."
    assert (
        round(abs(train_ratio) + abs(test_ratio) + abs(cv_ratio)) == 1.0
    ), "Ratios must be positive."

    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    cv_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, cv_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, cv_size]
    )
    return train_dataset, test_dataset, cv_dataset

from typing import Tuple, Union

import torch
from torchvision import datasets, transforms

from common import config


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE, transforms_: Union[None, list] = None
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Get dataloaders with MNIST train, test, and cross validation data sets.

    Keyword Arguments:
        batch_size {int} -- (default: {config.BATCH_SIZE})
        transforms_ {Union[None, list]} -- list of torch transforms (default: None)

    Returns:
        Tuple[torch.utils.data.DataLoader, ...] -- Train, test, and cross-validation
    """
    if transforms_ is None:
        transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

    transform = transforms.Compose(transforms_)

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

    Arguments:
        dataset {torch.utils.data.Dataset} -- all training data (features and labels)

    Keyword Arguments:
        train_ratio {float}
            -- Training data ratio to all data (default: {config.TRAIN_RATIO})
        test_ratio {float} -- Test data ratio to all data (default: {config.TEST_RATIO})
        cv_ratio {float} -- Cross-Validation data (default: {config.CV_RATIO})

    Returns:
        Tuple[torch.utils.data.Dataset, ...] -- train, test, and cross-validation

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

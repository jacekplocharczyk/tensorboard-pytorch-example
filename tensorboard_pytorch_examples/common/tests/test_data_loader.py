import pytest
from torchvision import datasets, transforms

from tensorboard_pytorch_examples.common import config, data_loader


@pytest.fixture
def example_dataset():
    dataset = datasets.MNIST(
        config.DATA_DIR / "mnist_train",
        train=True,
        download=True,
        transform=[transforms.ToTensor()],
    )
    dataset.data = dataset.data[:100, :, :]

    return dataset


def test_split_dataset_1(example_dataset):
    train_dataset, test_dataset, cv_dataset = data_loader.split_dataset(example_dataset)
    assert len(train_dataset) == 70
    assert len(test_dataset) == 15
    assert len(cv_dataset) == 15


def test_split_dataset_2(example_dataset):
    train_dataset, test_dataset, cv_dataset = data_loader.split_dataset(
        example_dataset, 0.6, 0.3, 0.1
    )
    assert len(train_dataset) == 60
    assert len(test_dataset) == 30
    assert len(cv_dataset) == 10


def test_split_dataset_3(example_dataset):
    with pytest.raises(AssertionError):
        data_loader.split_dataset(example_dataset, 2, 0, 0)
    with pytest.raises(AssertionError):
        data_loader.split_dataset(example_dataset, 2, -1, 0)


def test_get_dataloaders(example_dataset):
    data_loader.get_dataloaders()  # just check if it runs

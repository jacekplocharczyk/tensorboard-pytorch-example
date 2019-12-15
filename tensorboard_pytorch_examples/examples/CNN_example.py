import torch
import torch.nn as nn

from tensorboard_pytorch_examples.common import config, data_loader, tensorboard_logs
from tensorboard_pytorch_examples.common.default_convnet import ConvNet
from tensorboard_pytorch_examples.common.trainer import ClassificationTrainer


def main():
    device = config.DEVICE
    epochs = config.DEFAULT_EPOCHS_COUNT
    lr = config.LEARNING_RATE

    writer = tensorboard_logs.TensorboardWriter(device, lr)

    trainloader, testloader, cvloader = data_loader.get_dataloaders()
    writer.add_image_sample(trainloader)

    net = ConvNet(outputs=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    trainer = ClassificationTrainer(
        trainloader, cvloader, criterion, writer, epochs, device
    )

    net = trainer(net, optimizer)

    writer.close()


if __name__ == "__main__":
    # TODO: add data augmentation
    # TODO: add autoencoder to reduce dimension (rather different project)
    # TODO: add docstring at the top
    main()

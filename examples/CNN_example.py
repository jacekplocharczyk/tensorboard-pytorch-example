import torch
import torch.nn as nn

from common import config, data_loader, tensorboard_logs
from common.default_convnet import ConvNet
from common.training_loop import training_loop


def main():
    device = config.DEVICE
    epochs = config.DEFAULT_EPOCHS_COUNT
    lr = config.LEARNING_RATE

    writer = tensorboard_logs.TensorboardWriter(device, lr, epochs)

    trainloader, testloader, cvloader = data_loader.get_dataloaders()
    writer.add_image_sample(trainloader)

    net = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net = training_loop(
        net,
        trainloader,
        testloader,
        cvloader,
        optimizer=optimizer,
        criterion=criterion,
        writer=writer,
        epochs=epochs,
        device=device,
    )

    writer.close()


if __name__ == "__main__":
    # TODO: add data augmentation
    # TODO: add autoencoder to reduce dimension (rather different project)
    main()

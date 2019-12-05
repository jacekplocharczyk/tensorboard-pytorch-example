import torch.nn as nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from common import data_loader
from common.default_convnet import ConvNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")


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

    trainloader, testloader, cvloader = data_loader.get_dataloaders()
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

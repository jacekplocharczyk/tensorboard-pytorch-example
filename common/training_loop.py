import torch
from torch.utils.tensorboard import SummaryWriter

from common.config import DEVICE, CPU_DEVICE, DEFAULT_EPOCHS_COUNT


def training_loop(
    model: torch.nn,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    cvloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    writer: SummaryWriter,
    epochs: int = DEFAULT_EPOCHS_COUNT,
    device: torch.device = DEVICE,
) -> torch.nn:

    # TODO: add docstring
    # TODO: add tests?
    # TODO: add more metrics (cv test, cv loss, per class error rate)

    step = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch + 1)
        for i, data in enumerate(trainloader, 0):
            step += 1
            inputs, labels = data[0].to(device), data[1].to(device)

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

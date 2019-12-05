import torch
from torch.utils.tensorboard import SummaryWriter

from common.config import CPU_DEVICE, DEFAULT_EPOCHS_COUNT, DEVICE


def training_loop(
    model: torch.nn,
    trainloader: torch.utils.data.DataLoader,
    cvloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    writer: SummaryWriter,
    epochs: int = DEFAULT_EPOCHS_COUNT,
    device: torch.device = DEVICE,
) -> torch.nn:
    """
    Learn ML model using trainloader and test using cross-validation loader.

    Arguments:
        model {torch.nn} -- model to lear
        trainloader {torch.utils.data.DataLoader} -- training data
        cvloader {torch.utils.data.DataLoader} -- cross-validation data
        optimizer {torch.optim.Optimizer}
        criterion {torch.nn.modules.loss._Loss}
        writer {SummaryWriter}

    Keyword Arguments:
        epochs {int} -- number of epochs to learn (default: {DEFAULT_EPOCHS_COUNT})
        device {torch.device} -- GPU or CPU device (default: {DEVICE})

    Returns:
        torch.nn -- trained model
    """

    # TODO: add tests?
    # TODO: add more metrics (cv test, cv loss, per class error rate)

    step = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch + 1)  # TODO: Use logger instead
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

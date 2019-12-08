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
    # TODO: Change this to the class?

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
        eval_cv_stats(model, step, trainloader, criterion, writer, device),

    return model


def eval_cv_stats(
    model: torch.nn,
    step: int,
    cvloader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    writer: SummaryWriter,
    device: torch.device = DEVICE,
):
    """
    Save cross-validation statistics to the tensorboard logs.

    Arguments:
        model {torch.nn} -- model to lear
        step {int} -- number of optimizer steps
        cvloader {torch.utils.data.DataLoader} -- cross-validation data
        criterion {torch.nn.modules.loss._Loss}
        writer {SummaryWriter}

    Keyword Arguments:
        device {torch.device} -- GPU or CPU device (default: {DEVICE})
    """
    cv_acc = torch.tensor(0.0).to(device)
    cv_loss = torch.tensor(0.0).to(device)
    samples_no = float(len(cvloader.dataset))

    with torch.no_grad():
        model = model.eval()

        for data in cvloader:
            batch_size = len(data[0])  # last sample can have different items

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            batch_acc = (outputs.argmax(1) == labels).sum()
            batch_loss = criterion(outputs, labels) * batch_size

            cv_acc += batch_acc
            cv_loss += batch_loss

        cv_acc = (cv_acc / samples_no).to(CPU_DEVICE)
        cv_loss = (cv_loss / samples_no).to(CPU_DEVICE)

        writer.add_scalar("Loss/cv", cv_loss, step)
        writer.add_scalar("Acc/cv", cv_acc, step)

        model = model.train()

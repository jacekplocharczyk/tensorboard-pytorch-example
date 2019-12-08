from typing import Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from common.config import CPU_DEVICE, DEFAULT_EPOCHS_COUNT, DEVICE


class ClassificationTrainer:
    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        cvloader: torch.utils.data.DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        writer: SummaryWriter,
        epochs: int = DEFAULT_EPOCHS_COUNT,
        device: torch.device = DEVICE,
        train_stats_frequency: int = 10,
    ) -> None:
        """
        Basic class used for training.

        Arguments:
            trainloader {torch.utils.data.DataLoader} -- training data
            cvloader {torch.utils.data.DataLoader} -- cross-validation data
            criterion {torch.nn.modules.loss._Loss}
            writer {SummaryWriter}

        Keyword Arguments:
            epochs {int} -- number of epochs to learn (default: {DEFAULT_EPOCHS_COUNT})
            device {torch.device} -- GPU or CPU device (default: {DEVICE})
            train_stats_frequency {int} -- tensorboard train data update frequency
                (default: {10})
        """
        self.trainloader = trainloader
        self.cvloader = cvloader
        self.criterion = criterion
        self.writer = writer
        self.epochs = epochs
        self.device = device
        self.train_stats_frequency = train_stats_frequency
        self.step = 0
        self._first_run = True

    def __call__(self, *args, **kwargs):
        return self._training_loop(*args, **kwargs)

    def _training_loop(
        self, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> nn.Module:
        """
        Update the model by applying optimizer steps.
        Perform {self.epochs} number of iterations over whole dataset.

        Arguments:
            model {torch.nn} -- model to learn
            optimizer {torch.optim.Optimizer}

        Returns:
            nn.Module -- trained model
        """

        for epoch in range(self.epochs):
            print(epoch + 1)  # TODO: Use logger instead
            for batch in self.trainloader:
                model = self._optimizer_step(model, optimizer, batch)

            self.update_cv_stats(model)
            self._first_run = False

        return model

    def _optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Tuple[torch.tensor, torch.tensor],
    ) -> nn.Module:
        """
        Perform one optimizer step on the batch.

        Arguments:
            model {torch.nn} -- model to learn
            optimizer {torch.optim.Optimizer}
            batch {Tuple[torch.tensor, torch.tensor]} -- batch with features and targets

        Returns:
            nn.Module -- updated model
        """
        self.step += 1

        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if self.step % self.train_stats_frequency == 0:
            self.update_train_stats(loss, outputs, targets)

        return model

    def update_train_stats(
        self, loss: torch.tensor, outputs: torch.tensor, targets: torch.tensor
    ) -> None:
        """[summary]

        Arguments:
            loss {torch.tensor}
            outputs {torch.tensor}
            targets {torch.tensor}

        """
        acc = (outputs.argmax(1) == targets).sum() / float(targets.shape[0])
        self.writer.add_scalar("Loss/train", loss.to(CPU_DEVICE), self.step)
        self.writer.add_scalar("Acc/train", acc.to(CPU_DEVICE), self.step)

    def update_cv_stats(self, model: torch.nn) -> None:
        """
        Update cross-validation statistics in the tensorboard logs.

        Arguments:
            model {torch.nn} -- model to learn

        """
        cv_acc, cv_loss, outputs_dist, targets_dist = self._get_cv_stats(model)
        self.writer.add_scalar("Loss/cv", cv_loss, self.step)
        self.writer.add_scalar("Acc/cv", cv_acc, self.step)
        self.writer.add_histogram("Outputs/cv", outputs_dist, self.step)
        if self._first_run:
            self.writer.add_histogram("Outputs/cv", targets_dist, 0)

    def _get_cv_stats(self, model: torch.nn) -> Tuple[torch.tensor, ...]:
        """
        Collect CV data accuracy, loss, prediction distribution, and targets
        distribution.

        Arguments:
            model {torch.nn} -- model to learn

        Returns:
            Tuple[torch.tensor * 4] -- accuracy, loss, prediction distribution, and
                targets distribution.
        """
        cv_acc = torch.tensor(0.0).to(self.device)
        cv_loss = torch.tensor(0.0).to(self.device)
        samples_no = float(len(self.cvloader.dataset))
        outputs_dist = None
        targets_dist = None

        with torch.no_grad():
            model = model.eval()

            for inputs, targets in self.cvloader:
                batch_size = inputs.shape[0]  # last sample can have different items

                targets = targets.to(self.device)
                outputs = model(inputs.to(self.device))

                if outputs_dist is None and targets_dist is None:
                    outputs_dist = outputs.argmax(1).long()
                    targets_dist = targets.long()
                else:
                    outputs_dist = torch.cat([outputs_dist, outputs.argmax(1).long()])
                    targets_dist = torch.cat([targets_dist, targets.long()])

                cv_acc += (outputs.argmax(1) == targets).sum()
                cv_loss += self.criterion(outputs, targets) * batch_size

            cv_acc = (cv_acc / samples_no).to(CPU_DEVICE)
            cv_loss = (cv_loss / samples_no).to(CPU_DEVICE)
            outputs_dist = outputs_dist.to(CPU_DEVICE)
            targets_dist = targets_dist.to(CPU_DEVICE)

        model = model.train()
        return cv_acc, cv_loss, outputs_dist, targets_dist

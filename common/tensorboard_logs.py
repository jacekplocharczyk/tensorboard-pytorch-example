import datetime
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from common.config import TENSORBOARD_DIR


class TensorboardWriter(SummaryWriter):
    def __init__(self, device: torch.device, lr: float, ep: int, comment: str = ""):
        log_dir = self.get_log_dir(device, lr, ep, comment)
        super().__init__(log_dir=log_dir)

    @staticmethod
    def get_log_dir(device: torch.device, lr: float, ep: int, comment: str) -> Path:
        """
        Get tensorboard log dir based on params

        Arguments:
            device {torch.device} -- GPU or CPU device
            lr {float} -- learning rate
            ep {int} -- epochs number
            comment {str} -- user comment

        Returns:
            Path -- log dir
        """

        dev_str = str(device).split(":")[0]
        tensorboard_suffix = f"_dev{dev_str}_lr{lr}_ep{ep}" + comment
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        log_dir = TENSORBOARD_DIR / (current_time + "_" + tensorboard_suffix)
        return log_dir

    def add_image_sample(
        self, dataloader: torch.utils.data.DataLoader, description: str = "images"
    ):
        """
        Add sample image from the dataloader.

        Arguments:
            dataloader {torch.utils.data.DataLoader} -- image data

        Keyword Arguments:
            description {str} -- user description (default: {"images"})
        """
        images, _ = next(iter(dataloader))

        grid = torchvision.utils.make_grid(images)
        self.add_image(description, grid, 0)

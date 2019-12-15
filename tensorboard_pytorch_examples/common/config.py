from pathlib import Path

import torch

BATCH_SIZE = 64
CPU_DEVICE = torch.device("cpu")
CV_RATIO = 0.15
DATA_DIR = Path("tensorboard_pytorch_examples/data/")
DEFAULT_EPOCHS_COUNT = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
TENSORBOARD_DIR = Path("tensorboard_logs/")
TEST_RATIO = 0.15
TRAIN_RATIO = 0.7

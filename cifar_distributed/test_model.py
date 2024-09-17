import os
import time
import torch
import torch.utils
import torch.utils.data
import torchvision
import timm
from metaflow._vendor import click

from torchvision import transforms
from metrics_logger import MetricsLogger
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from metaflow.checkpoint_utils import Checkpoint


def test_model(
    batch_size,
    workers,
    img_dims,
    model_name,
    model_path,
):
    if model_path is None:
        raise ValueError("model_path is required")

    # Define data transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_dims),
        ]
    )
    CLASSES = 10
    torch.cuda.empty_cache()

    # Create model
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=CLASSES,
    )
    checkpoint = torch.load(model_path)
    print(
        "Loading model from checkpoint: ",
        model_path,
        "checkpoint keys",
        checkpoint["model_state_dict"].keys(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Load CIFAR10 dataset
    data = torchvision.datasets.CIFAR10(
        "./", train=False, download=True, transform=transform
    )

    test_data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    logger = MetricsLogger(
        {},
        save_steps=5,
    )
    _step = 0
    _val_step = 0
    model.eval()
    # Validation Loop comes here.
    val_losses = []
    val_accs = []
    for batch in tqdm(test_data_loader, total=len(test_data_loader)):
        features, labels = batch[0].to(device), batch[1].to(device)

        preds = model(features)
        loss = loss_fn(preds, labels)
        _valacc = (preds.argmax(1) == labels).float().mean().item()
        val_losses.append(loss.item())
        val_accs.append(_valacc)
        logger.log_step(_val_step, "test_loss", loss.item())
        logger.log_step(_val_step, "test_acc", _valacc)
        _val_step += 1

    overall_val_loss = sum(val_losses) / len(val_losses)
    overall_val_acc = sum(val_accs) / len(val_accs)
    return overall_val_loss, overall_val_acc

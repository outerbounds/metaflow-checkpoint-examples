import os
import tempfile
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
from metaflow import Checkpoint

IMAGE_DIMS = (336, 336)


def ensure_environment_is_set(*vars):
    for var in vars:
        if var not in os.environ:
            raise ValueError(
                "Missing environment variable to start the execution : %s " % var
            )


# One key reason for this abstraction is checkpoint saving can work out of the box with just
# `model.state_dict()` since the model is wrapped in `DDP` and when checkpoints are loaded
# the model is already wrapped in the DPP wrapper. But when models are loaded outside the training
# context (i.e. outside the distributed wrapper), we need to ensure that the model's original state
# is loaded and not the `DDP` wrapped model.
def save_model(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            # When we save model we need to ensure that it's stripping the `DDP` wrapper
            # this because the DDP wrapper will prefix the `module` to the model's state_dict
            # and when we load the model, we need to ensure that we are loading the model
            # in a way that works outside the distributed context
            "model_state_dict": model.module.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


@click.command()
@click.option("--batch-size", default=256, help="Batch size to use.")
@click.option("--epochs", default=5, help="Number of epochs to train.")
@click.option(
    "--data-loading-workers",
    "workers",
    default=4,
    help="Number of workers for data loading.",
)
@click.option("--img-dims", default=IMAGE_DIMS, help="Image dimensions for resizing.")
@click.option("--model-name", default="resnet50d", help="Model name to use.")
@click.option(
    "--checkpoint-load-path",
    default=None,
    help="Path to the checkpoint to resume from.",
)
@click.option(
    "--model-save-path", default="final_model.pth", help="Path to save the FINAL model."
)
@click.option("--from-scratch", type=int, default=0, help="Train from scratch.")
@click.option("--learning-rate", default=0.01, help="Learning rate for the optimizer.")
@click.option(
    "--best-is-latest", type=bool, default=False, help="Set best model as latest model."
)
def train_model(
    batch_size,
    epochs,
    workers,
    img_dims,
    model_name,
    checkpoint_load_path,
    model_save_path,
    from_scratch,
    learning_rate,
    best_is_latest,
):
    ensure_environment_is_set("WORLD_SIZE", "RANK", "LOCAL_RANK")
    # Initialize distributed process group
    dist.init_process_group("nccl")

    # Get local rank and global rank from environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(
        "Running CIFAR10 training on Global rank",
        global_rank,
        "Local Rank",
        local_rank,
        "World Size",
        world_size,
    )

    is_rank_zero = global_rank == 0

    # Adjust batch size based on world size
    batch_size = batch_size // world_size

    SAVE_STEPS = 20

    # Define data transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_dims),
        ]
    )

    ds_dir = tempfile.mkdtemp()
    # Load CIFAR10 dataset
    data = torchvision.datasets.CIFAR10(
        ds_dir, train=True, download=True, transform=transform
    )

    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    CLASSES = 10
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_data),
        num_workers=workers,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=DistributedSampler(val_data),
    )

    # Set device for the current process
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    # Create model
    optimizer_path = None
    model = timm.create_model(
        model_name,
        pretrained=not bool(from_scratch),
        num_classes=CLASSES,
    )

    model = model.to("cuda:" + str(local_rank))
    model = DDP(model, device_ids=[local_rank])

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if checkpoint_load_path is not None:
        file_name = "best_model.pth" if best_is_latest else "epoch.pth"
        checkpoint = torch.load(os.path.join(checkpoint_load_path, file_name))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        del checkpoint  # Free up memory
        print(
            "Loaded checkpoint from",
            checkpoint_load_path,
            "Starting from epoch",
            start_epoch,
        )

    logger = MetricsLogger(
        {
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_name": model_name,
            "img_dims": img_dims,
            "using_checkpoint": checkpoint_load_path is not None,
            "num_epochs": epochs,
            "training_from_scratch": from_scratch,
        },
        save_steps=SAVE_STEPS,
    )

    if optimizer_path is not None and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))

    # Start training
    checkpoint = Checkpoint(init_dir=True)
    best_path = os.path.join(checkpoint.directory, "best_model.pth")
    epoch_path = os.path.join(checkpoint.directory, "epoch.pth")

    start = time.perf_counter()
    best_val_loss = float("inf")
    _step = 0
    _val_step = 0
    for epoch in range(start_epoch, epochs):
        print("Starting Epoch", epoch)
        epoch_start_time = time.perf_counter()

        train_losses = []
        train_accs = []
        # Training Loop Comes here
        model.train()
        for batch in tqdm(train_data_loader, total=len(train_data_loader)):
            features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

            optimizer.zero_grad()
            preds = model(features)
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()
            logger.log_step(_step, "train_loss", loss.item())
            train_acc = (preds.argmax(1) == labels).float().mean().item()
            logger.log_step(_step, "train_acc", train_acc)
            train_losses.append(loss.item())
            train_accs.append(train_acc)
            _step += 1

        # Calculate the loss/accurary over here.
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        logger.log_epoch(epoch, "train_loss", train_loss)
        logger.log_epoch(epoch, "train_acc", train_acc)

        model.eval()
        # Validation Loop comes here.
        val_losses = []
        val_accs = []
        for batch in tqdm(val_data_loader, total=len(val_data_loader)):
            features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

            preds = model(features)
            loss = loss_fn(preds, labels)
            val_losses.append(loss.item())
            _valacc = (preds.argmax(1) == labels).float().mean().item()
            val_accs.append(_valacc)
            logger.log_step(_val_step, "val_loss", loss.item())
            logger.log_step(_val_step, "val_acc", _valacc)
            _val_step += 1

        # Calculate the loss/accurary over here.
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)
        logger.log_epoch(epoch, "val_loss", val_loss)
        logger.log_epoch(epoch, "val_acc", val_acc)
        logger.save()

        chckpt_metadata = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }

        # Save a model every epoch
        if is_rank_zero:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                epoch_path,
            )
            checkpoint.save(
                epoch_path,
                metadata=chckpt_metadata,
                name=f"epoch_checkpoint",
                latest=not best_is_latest,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    best_path,
                )
                checkpoint.save(
                    best_path,
                    metadata=chckpt_metadata,
                    name="best_model",
                    latest=best_is_latest
                    # We set latest over here because this is the checkpoint that we will
                    # assume will be reloaded onto the checkpoint directory.
                    # And since this has a static names, we can rely on it being statically
                    # present in the directory that got loaded
                )

        epoch_end_time = time.perf_counter()
        if global_rank == 0:
            print(f"Epoch {epoch+1} Time", epoch_end_time - epoch_start_time)

    end = time.perf_counter()
    if global_rank == 0:
        print("Training Took", end - start)
        if model_save_path is not None:
            print(
                "Saving the model to",
                model_save_path,
            )
            save_model(model, optimizer, epochs, model_save_path)


if __name__ == "__main__":
    train_model()

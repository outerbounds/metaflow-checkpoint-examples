import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.nn import Module
from torchvision import models
import torchmetrics
from metaflow import Checkpoint
from metaflow import current


class MNISTClassifier(Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 10),
        )
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    save_dir=None,
    start_epoch=0,
    epochs=10,
):
    ## METAFLOW-SPECIFIC-CODE
    checkpoint = current.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    latest_checkpoint = None
    if save_dir is None:
        os.makedirs("model_checkpoints", exist_ok=True)
        save_dir = "model_checkpoints"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    elif not os.path.isdir(save_dir):
        raise ValueError(f"{save_dir} is not a directory")

    best_loss, best_acc = float("inf"), 0
    best_path = os.path.join(save_dir, "best", "checkpoint.pth")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    print("starting training from epoch", start_epoch)
    for epoch in range(start_epoch, epochs):
        model.train()

        epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        checkpoint_epoch_path = os.path.join(epoch_dir, f"checkpoint.pth")

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

        val_loss, val_acc = validate_model(model, val_loader, device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            # Also write the best checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": loss,
                    "val_acc": val_acc,
                },
                best_path,
            )
            ## METAFLOW-SPECIFIC-CODE
            checkpoint.save(
                best_path,
                name="best",
            )

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "val_acc": val_acc,
            },
            checkpoint_epoch_path,
        )
        ## METAFLOW-SPECIFIC-CODE
        latest_checkpoint = checkpoint.save(
            epoch_dir,
        )

        if epoch % 2 == 0 and epoch > 0 and current.retry_count == 0:
            raise Exception("Simulating Failure")

    return best_loss, best_acc, latest_checkpoint


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct / len(val_loader.dataset)
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss,
            correct,
            len(val_loader.dataset),
            val_acc,
        )
    )
    return val_loss, val_acc


def test(checkpoint_path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = MNIST(os.getcwd(), download=True, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = MNISTClassifier()
    _checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(_checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    validate_model(model, test_loader, device)


def train(checkpoint_path=None, num_epochs=10, start_epoch=0, model_save_dir=None):
    # Transforms for the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # MNIST dataset
    dataset = MNIST(os.getcwd(), download=True, transform=transform, train=True)
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)

    model = MNISTClassifier()
    optimizer = Adam(model.parameters(), lr=1e-3)

    if checkpoint_path:
        _checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(_checkpoint["model_state_dict"])
        optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
        start_epoch = _checkpoint["epoch"]
        print(
            "Loading model from checkpoint with epoch",
            start_epoch,
            "and loss",
            _checkpoint["loss"],
            "and accuracy",
            _checkpoint["val_acc"],
        )

    return train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=num_epochs,
        start_epoch=start_epoch,
        save_dir=model_save_dir,
    )


if __name__ == "__main__":
    train()

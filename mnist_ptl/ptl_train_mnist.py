import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.optim import Adam
from torchmetrics import Accuracy


class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define model architecture
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 10),
        )
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        # Forward pass through the network
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("val_acc", self.accuracy, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Test step
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.log("test_acc", self.accuracy, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # Configure optimizers
        return Adam(self.parameters(), lr=1e-3)


def train_mnist(checkpoint_path=None, num_epochs=10, callbacks=[]):
    # Transforms for the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # MNIST dataset
    dataset = MNIST(os.getcwd(), download=True, transform=transform, train=True)
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64)
    val_loader = DataLoader(mnist_val, batch_size=64)

    # Load model from checkpoint if path is provided
    if checkpoint_path:
        model = MNISTClassifier.load_from_checkpoint(checkpoint_path)
    else:
        model = MNISTClassifier()

    # Trainer
    trainer = pl.Trainer(max_epochs=num_epochs, callbacks=callbacks)

    # Train the model
    trainer.fit(model, train_loader, val_loader)


def test_mnist(checkpoint_path):
    # Transforms for the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # MNIST dataset
    dataset = MNIST(os.getcwd(), download=True, transform=transform, train=False)
    # mnist_test, _ = random_split(dataset, [10000, 60000])

    test_loader = DataLoader(dataset, batch_size=64)

    # Load model from checkpoint
    model = MNISTClassifier.load_from_checkpoint(checkpoint_path)

    # Trainer
    trainer = pl.Trainer()

    # Test the model
    result = trainer.test(model, dataloaders=test_loader)
    print(result)


if __name__ == "__main__":
    # Specify the path to your checkpoint file here
    checkpoint_path = None  # 'path/to/your/checkpoint.ckpt'
    # If you want to train from scratch, just pass None or don't pass any argument to train_mnist()
    train_mnist(checkpoint_path)

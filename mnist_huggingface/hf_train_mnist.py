from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn
from safetensors.torch import load_file as safe_load_tensor_file
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_metric


# Define a PyTorch dataset
class PyTorchMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return {"pixel_values": image, "labels": label}


# Model definition
class MNISTModel(nn.Module):
    def __init__(self, num_labels=10):
        super(MNISTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values.float())

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def load_train_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset_mnist = MNIST(".", download=True, transform=transform, train=True)
    train_images = train_dataset_mnist.data.numpy()
    train_labels = train_dataset_mnist.targets.numpy()
    train_images = train_images.reshape(train_images.shape[0], -1)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.1
    )
    train_dataset = PyTorchMNISTDataset(train_images, train_labels)
    val_dataset = PyTorchMNISTDataset(val_images, val_labels)
    return train_dataset, val_dataset


def load_test_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset_mnist = MNIST(".", download=True, transform=transform, train=False)
    test_images = test_dataset_mnist.data.numpy()
    test_labels = test_dataset_mnist.targets.numpy()
    test_images = test_images.reshape(test_images.shape[0], -1)
    test_dataset = PyTorchMNISTDataset(test_images, test_labels)
    return test_dataset


# Training and testing functions
def train(
    checkpoint_path=None,
    num_epochs=3,
    callbacks=[],
    output_dir="./results",
    logging_dir="./logs",
):
    train_dataset, val_dataset = load_train_data()

    model = MNISTModel()
    if checkpoint_path is not None:
        model.load_state_dict(safe_load_tensor_file(checkpoint_path))

    training_args = TrainingArguments(
        log_level="debug",
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.evaluate()
    print("completed training!!!")


def test(checkpoint_path):
    test_dataset = load_test_data()
    model = MNISTModel()
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        logging_steps=10,
    )

    model.load_state_dict(safe_load_tensor_file(checkpoint_path))

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    test_metrics = trainer.evaluate()
    print(test_metrics)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load_metric("accuracy", trust_remote_code=True).compute(
        predictions=predictions, references=labels
    )


if __name__ == "__main__":
    train()

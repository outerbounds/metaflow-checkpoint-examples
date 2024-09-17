from metaflow import FlowSpec, step, Parameter, current, checkpoint, model
from typing import List


class MnistTorchFlow(FlowSpec):

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @checkpoint
    @step
    def train(self):
        from torch_train_mnist import train

        self.best_loss, self.best_acc, self.latest_checkpoint = train(
            num_epochs=self.epochs,
            model_save_dir="./model_checkpoints/" + current.task_id,
        )
        self.next(self.test)

    @model(load=["latest_checkpoint"])
    @step
    def test(self):
        import os
        from torch_train_mnist import test

        test(os.path.join(current.model.loaded["latest_checkpoint"], "best_model.pth"))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MnistTorchFlow()

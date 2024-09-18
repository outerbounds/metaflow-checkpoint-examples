from metaflow import FlowSpec, step, Parameter, current, checkpoint, model, retry
from typing import List
import os


class MnistTorchFlow(FlowSpec):

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @retry(times=3)
    @checkpoint
    @step
    def train(self):
        from torch_train_mnist import train

        checkpoint_path = None
        start_epoch = 0
        if current.checkpoint.is_loaded:
            # When users can call `current.checkpoint.save()` or `Checkpoint().save()`,
            # it informs Metaflow of what to load as a checkpoint in case of a failure.
            # The @checkpoint decorator automatically loads the latest checkpoint's
            # directory in the `current.checkpoint.directory`. Users can then deal with
            # the checkpoint path as they see fit.
            checkpoint_path = os.path.join(
                current.checkpoint.directory, "best_model.pth"
            )
            start_epoch = int(
                open(
                    os.path.join(
                        current.checkpoint.directory, "checkpoint_epoch_number"
                    )
                ).read()
            )

        self.best_loss, self.best_acc, self.latest_checkpoint = train(
            checkpoint_path=checkpoint_path,
            num_epochs=self.epochs,
            model_save_dir="./model_checkpoints/" + current.task_id,
            start_epoch=start_epoch,
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

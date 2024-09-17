from metaflow import FlowSpec, step, Parameter, checkpoint, current, model
import sys


class MNISTClassifierFlow(FlowSpec):

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @checkpoint
    @step
    def train(self):
        from ptl_train_mnist import train_mnist
        from ptl_callback import MetaflowPTLCheckpoint
        import os

        checkpointdir = current.checkpoint.directory
        # Checkpoint callback to save the best model based on validation loss
        best_cb = MetaflowPTLCheckpoint(
            monitor="val_loss",
            dirpath=checkpointdir,
            filename="loss-checkpoint-{epoch}-{step}",
            save_top_k=1,  # Save only the best model
            mode="min",  # Minimize validation loss
            name="val_loss_best",
        )
        acc_cb = MetaflowPTLCheckpoint(
            monitor="val_acc",
            dirpath=checkpointdir,
            filename="acc-checkpoint-{epoch}-{step}",
            save_top_k=3,
            mode="max",  # max validation loss
            name="val_acc_top5",
        )
        checkpoint_path = None

        if current.checkpoint.is_loaded:
            filename = current.checkpoint.info.metadata["file_names"][0]
            checkpoint_path = os.path.join(checkpointdir, filename)

        train_mnist(
            checkpoint_path=checkpoint_path,
            callbacks=[best_cb, acc_cb],
            num_epochs=self.epochs,
        )
        self.best_checkpoint = best_cb.latest_checkpoint
        self.file_name = os.path.basename(best_cb.best_model_path)
        print(self.file_name)
        self.next(self.test)

    @model(
        load=["best_checkpoint"],
    )
    @step
    def test(self):
        import os
        from ptl_train_mnist import test_mnist

        test_mnist(
            os.path.join(current.model.loaded["best_checkpoint"], self.file_name)
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MNISTClassifierFlow()

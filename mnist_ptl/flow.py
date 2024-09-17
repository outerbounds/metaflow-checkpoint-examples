from metaflow import FlowSpec, step, Parameter, checkpoint, current, model
import sys


class MNISTClassifierFlow(FlowSpec):

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @model
    @checkpoint
    @step
    def train(self):
        from ptl_train_mnist import train_mnist
        from ptl_callback import MetaflowCheckpoint
        import os

        checkpointdir = current.checkpoint.directory
        # MetaflowCheckpoint callback uses the `Checkpoint`/`current.checkpoint`
        # API to checkpoint models from Pytorch Lightning

        # Checkpoint callback to save the best model based on validation loss
        best_cb = MetaflowCheckpoint(
            monitor="val_loss",
            dirpath=checkpointdir,
            filename="loss-checkpoint-{epoch}-{step}",
            save_top_k=1,  # Save only the best model
            mode="min",  # Minimize validation loss
            name="val_loss_best",
            set_latest=True,
        )
        # Checkpoint callback to save the top 3 models based on validation accuracy
        acc_cb = MetaflowCheckpoint(
            monitor="val_acc",
            dirpath=checkpointdir,
            filename="acc-checkpoint-{epoch}-{step}",
            save_top_k=3,
            mode="max",  # max validation loss
            name="val_acc_top5",
            set_latest=False,
        )
        checkpoint_path = None

        # If the checkpoint is loaded from a previous run,
        # ensure that the checkpoint's path is passed to the
        # training function
        if current.checkpoint.is_loaded:
            # The MetaflowCheckpoint Pytorch Lightning callback
            # saves the best model's file-name in the checkpoint's metadata
            # which can be used to construct the checkpoint's path
            filename = current.checkpoint.info.metadata["file_names"][0]
            checkpoint_path = os.path.join(checkpointdir, filename)

        train_mnist(
            checkpoint_path=checkpoint_path,
            callbacks=[best_cb, acc_cb],
            num_epochs=self.epochs,
        )

        # Pytorch Ligthing's ModelCheckpoint callback
        # exposes the best model path as best_model_path
        best_model_path = best_cb.best_model_path
        self.file_name = os.path.basename(best_model_path)

        # Save the best model's reference as a Metaflow
        # data artifact. This reference can be used later
        # in the flow to load the model and perform inference
        # These artifacts created by @model/@checkpoint can be
        # loaded via the `@model` decorator's load parameter
        self.trained_model = current.model.save(
            best_model_path,
            label="best_model",
            metadata={
                "file_name": self.file_name,
                "score": best_cb.best_model_score.tolist()
                if best_cb.best_model_score is not None
                else None,
            },
        )

        self.next(self.test)

    @model(
        load="trained_model",
    )
    @step
    def test(self):
        import os
        from ptl_train_mnist import test_mnist

        # `current.model.loaded` offers and interface that will allow
        # accessing loaded model's paths. What ever is saved
        test_mnist(os.path.join(current.model.loaded["trained_model"], self.file_name))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MNISTClassifierFlow()

from json import load
from metaflow import FlowSpec, step, Parameter, current, checkpoint, model


class MNISTClassifierKerasFlow(FlowSpec):
    """
    Simple Train/Test flow with Keras and @checkpoint.
    """

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @model
    @checkpoint
    @step
    def train(self):
        from keras_train_mnist import train
        from keras_callback import MetaflowKerasCheckpoint
        import os

        # Checkpoint callback to save the best model based on validation loss
        best_cb = MetaflowKerasCheckpoint(
            filepath="./model_checkpoints/"
            + current.task_id
            + "/checkpoint_{epoch}.keras",
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        model = train(callbacks=[best_cb], num_epochs=self.epochs)
        self.best_checkpoint = best_cb.latest_checkpoint
        model.save("keras_model.h5")
        self.keras_model = current.model.save(
            "keras_model.h5",
            metadata={
                "epochs": self.epochs,
            },
        )
        self.filename = "keras_model.h5"
        self.next(self.test)

    @model(load=["keras_model"])
    @step
    def test(self):
        from keras_train_mnist import test
        import os

        test(os.path.join(current.model.loaded["keras_model"], "keras_model.h5"))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MNISTClassifierKerasFlow()

from metaflow import FlowSpec, step, Parameter, current, checkpoint, model


class MnistHFTrainerFlow(FlowSpec):

    epochs = Parameter("epochs", default=8, help="Number of epochs to train the model")

    @step
    def start(self):
        self.next(self.train)

    @checkpoint
    @step
    def train(self):
        from hf_train_mnist import train
        from hf_trainer_callback import MetaflowCheckpointCallback
        import os
        from transformers.trainer_callback import PrinterCallback

        best_cb = MetaflowCheckpointCallback()
        train(
            callbacks=[best_cb, PrinterCallback()],
            num_epochs=self.epochs,
        )
        self.best_checkpoint = best_cb.latest_checkpoint
        self.next(self.test)

    @model(load="best_checkpoint")
    @step
    def test(self):
        from hf_train_mnist import test
        import os

        test(
            os.path.join(
                current.model.loaded["best_checkpoint"],
                "model.safetensors",
            )
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    MnistHFTrainerFlow()

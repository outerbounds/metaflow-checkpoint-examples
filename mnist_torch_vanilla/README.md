# Training Handwriting Recognition on MNIST Dataset with Pytorch

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

```bash
python flow.py run --epochs 10
```

## Salient Features 

- **Automatic Checkpoint Loading on Failure**: This [flow](./flow.py) uses the `@checkpoint` functionality with a vanilla pytorch training loop. The `@checkpoint` decorator additonally exposes a `Checkpoint` object that can be imported from metaflow like `from metaflow import Checkpoint` This object can be use to save (and even load) checkpoints during a `@step`'s execution. This flow intentionally causes a failure so that training can be resumed from the last checkpoint. The below logic in the code handles the checkpoint reloading in the event of a failure. 
    ```python
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
        # The checkpoint path is only set if 
        # a checkpoint is loaded from a previous execution
        checkpoint_path=checkpoint_path,
        num_epochs=self.epochs,
        model_save_dir="./model_checkpoints/" + current.task_id,
        start_epoch=start_epoch,
    )
    ```
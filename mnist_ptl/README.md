

# Training Handwriting Recognition on MNIST Dataset with Pytorch Lightning

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

- **Metaflow Checkpointing with Pytorch Lightning Callbacks**: This [flow](./flow.py) uses the `@checkpoint` functionality in tandem withe [Pytorch Lightning's ModelCheckpoint mechanism](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html). The [MetaflowCheckpoint class](./ptl_callback.py) can be plugged into any Pytorch Lightning model called within a Metaflow flow execution to enable checkpointing. The following code block in the flow helps ensure that checkpoints are reloaded in the event of a failure. 
    ```python
    # If the checkpoint is loaded from a previous run,
    # ensure that the checkpoint's path is passed to the
    # training function
    if current.checkpoint.is_loaded:
        # The MetaflowCheckpoint Pytorch Lightning callback
        # saves the best model's file-name in the checkpoint's metadata
        # which can be used to construct the checkpoint's path
        filename = current.checkpoint.info.metadata["file_names"][0]
        checkpoint_path = os.path.join(checkpointdir, filename)
    ```
- **Restarting Previous Executions**: The flow can be restarted from the last checkpoint by setting the `@checkpoint(load_policy="eager")` on the step. This will ensure that the flow will start from the last checkpoint created by any previous execution. This allows users to start entirely new executions from the last checkpoint.
- **Model Loading in subsequent `@step`s**: This flow showcases a pattern of saving models via the `current.model.save` interface. Calling this API returns a reference object that can be set as a [data artifact in Metaflow](https://docs.metaflow.org/metaflow/basics#artifacts) and then be loaded in subsequent steps via `@model`. The [flow.py](./flow.py) demonstrates how to load the checkpoint in the `test` step by calling `@model(load="trained_model")`.
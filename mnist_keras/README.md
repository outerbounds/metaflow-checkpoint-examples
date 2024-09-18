# Training Handwriting Recognition on MNIST Dataset with Keras

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

```bash
python flow.py run  
```


## Salient Features 

- **Metaflow Checkpointing with Keras Callbacks**: This [flow](./flow.py) uses the `@checkpoint` functionality in tandem withe [Keras's callback mechanism](https://keras.io/api/callbacks/). The [MetaflowKerasCheckpoint class](./keras_callback.py) can be plugged into any Keras model called within a Metaflow flow execution to enable checkpointing.
- **Checkpoint Loading in subsequent `@step`s**: The callback exposese the `lastest_checkpoint` reference object which is returned by `Checkpoint().save()` or `current.checkpoint.save()`. This reference object can be set as a [data artifact in Metaflow](https://docs.metaflow.org/metaflow/basics#artifacts) and then be loaded in subsequent steps. The [flow.py](./flow.py) demonstrates how to load the checkpoint in the `test` step by calling `@model(load="best_checkpoint")`.
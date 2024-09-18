# Training LoRA's with Huggingface and Peft

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

```bash
python remote_flow.py run --config-file experiment_config.yaml
```


## Salient Features 

- **Loading and caching models from Huggingface**: This [flow](./remote_flow.py) uses the `@huggingface_hub` decorator to load and cache models from Huggingface. This allows uses to pull models from external sources like `huggingface_hub` and avoid any unncessary re-downloads. `@huggingface_hub` injects methods to `current` like `current.huggingface_hub.snapshot_download`. `current.huggingface_hub.snapshot_download` downloads the model from the Huggingface Hub and saves it in the backend storage based on the model's `repo_id`. If there exists a model with the same `repo_id` in the backend storage, it will not download the model again. The return value of the function is a reference to the model in the backend storage. This reference can become a first class data artifact in Metaflow and then be used to load the models in the subsequent `@step`s via `@model(load=["hf_model_checkpoint"])`

- **Automatic Checkpoint Reloading on Failure**: This [flow](./remote_flow.py) integrates `@checkpoint` with Huggingface's `Trainer`'s callback mechanism similar to the [example that trains handwriting recognition with the MNIST dataset](../mnist_huggingface/). This example showcases how the checkpoint can be reloaded in cases or failures by just using this simple block of code that derives if the checkpoint needs to be used by the underlying function.    
```python
self.config.model.resuming_checkpoint_path = None
if current.checkpoint.is_loaded:
    # Checkpoints Saved via the `MetaflowCheckpointCallback`
    # will be automatically loaded on retries so we just need to pass the 
    # underlying function the path where the checkpoint was loaded
    self.config.model.resuming_checkpoint_path = current.checkpoint.directory

```
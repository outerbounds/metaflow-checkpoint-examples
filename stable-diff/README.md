# Generating Videos From Text using Stable Diffusion and Metaflow

[This flow](./text_to_video_with_chkpt.py) generates videos from text as a multi `@step` process. It first downloads and caches all models from `huggingface_hub` via `@huggingface_hub` and the uses one model to create the images from the prompts. The images are the saved via the usage of `@model` and the passed down to the next step to create a video using the other models it downloaded. Finally the videos are painted into a [Metaflow card](https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-components). 

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

```bash
python text_to_video_with_chkpt.py run --config-file video_config.yaml
```

## Salient Features

- **Downloading and caching multiple models from Huggingface**: This flow will download and cache multiple models from Huggingface Hub using `@huggingface_hub` decorator.
- **Using the `@checkpoint` functionality to create a cache**: The `@checkpoint` decorator exposes a way to list all previous checkpoints created by previous executions of a `@step`. This can be useful to retrieve values that have been pre-computed in previous runs. The following code snippet in the code is responsible for caching: 
    ```python
    def _sdxl_checkpoint_from_cache(self):
        checkpoints = list(
            current.checkpoint.list(name="sdxl_model_pipeline", within_task=False)
        )
        if len(checkpoints) > 0:
            return checkpoints[0]
        self._save_sdxl_model(current.checkpoint.directory)
        return current.checkpoint.save(name="sdxl_model_pipeline")

    ```
- **Composing Multiple Models as a part of the execution**: This flow uses multiple models to create the final video all loaded at different points in the execution. This reduces the overhead of the user's machine to hold all the models at the same time along with making it very efficient when running the code [remotely](https://docs.metaflow.org/scaling/remote-tasks/introduction) or [automously](https://docs.metaflow.org/production/introduction). 
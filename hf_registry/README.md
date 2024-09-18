# Working with Models from external source in Metaflow like Hugging Face Hub

This is a simple example that showcases how users can leverage the `@huggingface_hub` decorator. The decorator exposes a `current.huggingface_hub.snapshot_download` function which wraps over [Huggingface's snapshot_download](https://huggingface.co/docs/huggingface_hub/v0.25.0/en/package_reference/file_download#huggingface_hub.snapshot_download) function. Calling the `current.huggingface_hub.snapshot_download` function will download the model from huggingface if it is not present in the Metaflow's datastore. If the model is present, it will just return a reference to the model so that it can be loaded in subsequent steps with the `@model` decorator.

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

- OSS Metaflow
    ```bash
    python flow.py --environment=pypi run --model-name stabilityai/stable-video-diffusion-img2vid --allow-patterns "svd.safetensors"
    ```

- Outerbounds Platform
    ```bash
    python flow.py --environment=fast-bakery run --model-name stabilityai/stable-video-diffusion-img2vid --allow-patterns "svd.safetensors"
    ```

## Salient Features 

- **Efficient Model Saving/Loading**: The `@huggingface_hub` decorator allows you to load models from Hugging Face Hub and cache them for increased performance benefits. It also ensures that models are versioned and managed appropriately in multi-user environments.

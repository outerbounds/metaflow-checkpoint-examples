# Distributed Training of Vision Models on CIFAR-10 with PyTorch DDP

This examples showcases how users can train a vision model on CIFAR-10 using PyTorch's Distributed Data Parallel (DDP) (multi-node). The flow uses the `@checkpoint` decorator to save the model's state periodically so that it can reload it when the execution gets interrupted. It also showcases how the model can then be used for testing in subsequent steps.

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow

- OSS Metaflow
    ```bash
    python flow.py --environment=pypi run --cluster-size 4 
    ```

- Outerbounds Platform
    ```bash
    python flow.py --environment=fast-bakery run --cluster-size 4
    ```

## Salient Features 

- **Gang Scheduled Distributed Training**: The flow usese the `@torchrun` and `@kubernetes` decorators together which allows [gang scheduling](https://en.wikipedia.org/wiki/Gang_scheduling) a bunch of jobs in a Kubernetes cluster. This example will even work with the [@batch decorator in Metaflow](https://docs.metaflow.org/api/step-decorators/batch). Users just need to switch the decorator from `@kubernetes` to `@batch`.

- **Automatic Failure Recovery**: With the `@checkpoint` decorator, the flow will automatically resume execution from the last checkpoint in case of failure. The below code block in the [flow.py](./flow.py) file demonstrates how users can choose to load a checkpoint in a flow. The checkpoint will be loaded from the "main node" (the node writing the checkpoints) and all worker nodes in the training cluster will load the same checkpoint. 
    ```python
    if current.checkpoint.is_loaded and "epoch" in current.checkpoint.info.metadata:
        print(
            "Using checkpoint from the execution", current.checkpoint.info.pathspec
        )
        checkpoint_load_dict = {
            "checkpoint-load-path": current.checkpoint.directory,
        }
    ```
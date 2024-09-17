packages = {
    "torch": "2.2.0",
    "torchvision": "0.17",
    "timm": "1.0.9",
    "tqdm": "4.66.5",
    "setuptools": "74.1.2",
    "numpy": "1.21.2",
}

import os
from metaflow import (
    current,
    FlowSpec,
    kubernetes,
    step,
    Parameter,
    torchrun,
    checkpoint,
    model,
    pypi_base,
    pypi,
    retry,
)
from metaflow.profilers import gpu_profile
from metrics_logger import metrics_logger

NUM_GPUS_PER_NODE = 1

# @pypi_base(python="3.12", packages=packages)
class Cifar10TestFlow(FlowSpec):

    model_key = Parameter(
        "model-key",
        type=str,
        help="Key to the trained model",
        default="mf.models_002/models/artifacts/cifar10_resnet50d_1f27db7c1228479e809025c28f3c0911",
    )

    batch_size = Parameter("batch-size", type=int, help="batch size", default=64)

    @step
    def start(self):
        self.next(self.test_model)

    @pypi(python="3.10", packages=packages)
    @kubernetes(
        cpu=8,
        memory=16000,
        gpu=1,
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.5",
        shared_memory=8000,
    )
    @metrics_logger
    @model(load="model_key")
    @step
    def test_model(self):
        from test_model import test_model
        from train_ddp import IMAGE_DIMS

        print(os.listdir(current.model.loaded["model_key"]))
        final_model_path = os.path.join(
            current.model.loaded["model_key"],
            "final_model.pth",
        )
        if not os.path.exists(final_model_path):
            raise ValueError("Model path does not exist")
        test_model(
            self.batch_size,
            4,
            IMAGE_DIMS,
            current.model.loaded.info["model_key"]["metadata"]["model-name"],
            final_model_path,
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Cifar10TestFlow()

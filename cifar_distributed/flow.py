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
    project,
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
from parallel_card import parallel_card

NUM_GPUS_PER_NODE = 1


@project(name="dist_testing")
class Cifar10DistributedTrainingFlow(FlowSpec):

    model_name = Parameter(
        "model-name", type=str, help="name of the model", default="resnet50d"
    )

    batch_size = Parameter("batch-size", type=int, help="batch size", default=64)

    test_batch_size = Parameter(
        "test-batch-size", type=int, help="test batch size", default=16
    )

    epochs = Parameter("epochs", type=int, help="number of epochs", default=5)

    learning_rate = Parameter(
        "learning-rate", type=float, help="learning rate", default=0.01
    )

    train_from_scratch = Parameter(
        "train-from-scratch", type=int, help="use pretrained model", default=0
    )

    best_is_latest = Parameter(
        "best-is-latest",
        type=bool,
        help="Set best model as latest model",
        default=False,
    )

    cluster_size = Parameter(
        "cluster-size",
        type=int,
        help="Number of nodes in the gang scheduled cluster",
        default=4,
    )

    @step
    def start(self):
        self.next(self.train, num_parallel=self.cluster_size)

    @parallel_card
    @retry(times=4)
    @metrics_logger
    @pypi(python="3.10", packages=packages)
    @gpu_profile(interval=1)
    @kubernetes(
        cpu=8,
        memory=16000,
        gpu=NUM_GPUS_PER_NODE,
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.5",
        shared_memory=8000,
    )
    @model
    @checkpoint
    @torchrun(all_nodes_started_timeout=60 * 60)  # 60 minutes
    @step
    def train(self):

        # some logic that deals with how we wish to reload the checkpoint
        # from the training run
        import torch

        checkpoint_load_dict = {}
        final_model_path = "final_model.pth"
        if current.checkpoint.is_loaded and "epoch" in current.checkpoint.info.metadata:
            print(
                "Using checkpoint from the execution", current.checkpoint.info.pathspec
            )
            checkpoint_load_dict = {
                "checkpoint-load-path": current.checkpoint.directory,
            }

        input_dict = {
            "epochs": self.epochs,
            "batch-size": self.batch_size,
            "learning-rate": self.learning_rate,
            "model-name": self.model_name,
            "model-save-path": final_model_path,
            "best-is-latest": str(self.best_is_latest),
        }
        input_dict.update(checkpoint_load_dict)

        if self.train_from_scratch:
            input_dict["from-scratch"] = 1

        current.torch.run(
            entrypoint="train_ddp.py",
            entrypoint_args=input_dict,
            nproc_per_node=NUM_GPUS_PER_NODE,
        )

        if current.parallel.node_index == 0:
            self.cifar_train_model = current.model.save(
                final_model_path,
                label="cifar10_%s" % self.model_name,
                metadata={
                    "epochs": self.epochs,
                    "batch-size": self.batch_size,
                    "learning-rate": self.learning_rate,
                    "model-name": self.model_name,
                },
            )
        self.next(self.join)

    @step
    def join(self, inputs):
        for i in inputs:
            if getattr(i, "cifar_train_model", None):
                self.cifar_train_model = i.cifar_train_model
                break
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
    @model(load="cifar_train_model")
    @step
    def test_model(self):
        from test_model import test_model
        from train_ddp import IMAGE_DIMS

        final_model_path = os.path.join(
            current.model.loaded["cifar_train_model"],
            "final_model.pth",
        )
        if not os.path.exists(final_model_path):
            raise ValueError("Model path does not exist")
        test_model(
            self.test_batch_size,
            4,
            IMAGE_DIMS,
            self.model_name,
            final_model_path,
        )
        self.next(self.end)

    @step
    def end(self):
        print(self.cifar_train_model)


if __name__ == "__main__":

    Cifar10DistributedTrainingFlow()

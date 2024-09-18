from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    current,
    card,
    project,
    huggingface_hub,
    checkpoint_config,
    checkpoint,
    model,
    retry,
    pypi,
)
import os

import tempfile
from metaflow import Checkpoint
from metaflow.profilers import gpu_profile

# HF_IMAGE = "006988687827.dkr.ecr.us-west-2.amazonaws.com/llm/hf-lora-pt:latest"
HF_IMAGE = "registry.hub.docker.com/valayob/hf-transformer-gpu:4.39.3.1"

CUDA_BASE_IMAGE = "nvcr.io/nvidia/pytorch:22.08-py3"

base_libraries = {
    "accelerate": "0.28.0",
    "appdirs": "1.4.4",
    "loralib": "0.1.2",
    "bitsandbytes": "0.43.0",
    "datasets": "2.18.0",
    # "fire": "0.6.0",
    "peft": "0.10.0",
    "transformers": "4.39.3",
    "sentencepiece": "0.2.0",
    "gradio": "4.25.0",
    "protobuf": "5.26.1",
    "torch": "2.2.2",
    "tokenizers": "0.15.2",
    "omegaconf": "2.4.0.dev3",
}


@checkpoint_config
@project(name="chkpt_lora")
class CudaPypiTestFlow(FlowSpec):
    @card
    @huggingface_hub
    @step
    def start(self):
        base_model = "openai/clip-vit-large-patch14"
        self.hf_model_checkpoint = current.huggingface_hub.snapshot_download(
            repo_id=base_model,
            ignore_patterns=["*.bin", "*.h5"],
        )
        self.next(self.end)

    @model(load=["hf_model_checkpoint"])
    @step
    def end(self):
        print(
            "Model was loaded on the path,", current.model.loaded["hf_model_checkpoint"]
        )
        print("Completed!")


if __name__ == "__main__":
    CudaPypiTestFlow()

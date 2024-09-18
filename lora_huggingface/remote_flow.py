from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    current,
    card,
    project,
    huggingface_hub,
    checkpoint,
    retry,
    model,
    pypi,
)
import os
from mixins import HuggingFaceLora, N_GPU, visible_devices

import tempfile
from metaflow.profilers import gpu_profile

# HF_IMAGE = "006988687827.dkr.ecr.us-west-2.amazonaws.com/llm/hf-lora-pt:latest"
HF_IMAGE = "registry.hub.docker.com/valayob/hf-transformer-gpu:4.39.3.1"


@project(name="chkpt_lora")
class LlamaInstructionTuning(FlowSpec, HuggingFaceLora):
    @card
    @huggingface_hub
    @step
    def start(self):
        base_model = self.config.model.base_model
        # `current.huggingface_hub.snapshot_download` downloads the model from the Hugging Face Hub
        # and saves it in the backend storage based on the model's `repo_id`. If there exists a model
        # with the same `repo_id` in the backend storage, it will not download the model again. The return
        # value of the function is a reference to the model in the backend storage.
        # This reference can be used to load the model in the subsequent steps via `@model(load=["hf_model_checkpoint"])`
        self.hf_model_checkpoint = current.huggingface_hub.snapshot_download(
            repo_id=base_model,
            ignore_patterns=[
                "*.bin",
            ],
        )
        current.card.extend(self.config_report())
        self.next(self.finetune)

    # @pypi(disabled=True)
    @card(customize=True)
    @gpu_profile(interval=0.5)
    @model(load=["hf_model_checkpoint"])
    @checkpoint
    @kubernetes(image=HF_IMAGE, gpu=N_GPU, cpu=14, memory=72000)
    @retry(times=3)
    @step
    def finetune(self):
        self.config.model.model_save_directory = current.checkpoint.directory

        self.config.model.resuming_checkpoint_path = None
        if current.checkpoint.is_loaded:
            # Checkpoints Saved via the `MetaflowCheckpointCallback`
            # will be automatically loaded on retries so we just need to pass the
            # underlying function the path where the checkpoint was loaded from.
            self.config.model.resuming_checkpoint_path = current.checkpoint.directory

        self.run(
            base_model_path=current.model.loaded["hf_model_checkpoint"],
            dataset_path=None,
        )

        current.card.extend(self.config_report())
        self.next(self.end)

    @step
    def end(self):
        print("Completed!")


if __name__ == "__main__":
    LlamaInstructionTuning()

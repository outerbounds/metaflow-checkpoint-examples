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


@checkpoint_config
@project(name="chkpt_lora")
class LlamaInstructionTuning(FlowSpec, HuggingFaceLora):

    # @kubernetes(cpu=4, memory=16000, disk=80000, image=HF_IMAGE)
    # @pypi(packages={"huggingface-hub":"0.16.4", "omegaconf":"2.4.0.dev3"}, python="3.11.5")
    @card
    @huggingface_hub
    @step
    def start(self):
        base_model = self.config.model.base_model
        self.hf_model_checkpoint = current.huggingface_hub.snapshot_download(
            repo_id=base_model,
            ignore_patterns=[
                "*.bin",
            ],
        )
        current.card.extend(self.config_report())
        self.next(self.finetune)

    # @pypi(disabled=True)
    # TODO : Use @model to load the checkpoint.
    @gpu_profile(interval=0.5)
    @model(load=["hf_model_checkpoint"])
    @checkpoint
    @kubernetes(image=HF_IMAGE, gpu=N_GPU, cpu=14, memory=72000)
    @retry(times=3)
    @step
    def finetune(self):
        self.config.model.model_save_directory = current.checkpoint.directory
        # Set the resuming checkpoint if there is a checkpoint loaded in the current context.
        self.config.model.resuming_checkpoint_path = None
        if (
            current.checkpoint.is_loaded
            and "latest_dir" in current.checkpoint.info.metadata
        ):
            self.config.model.resuming_checkpoint_path = os.path.join(
                current.checkpoint.directory,
                current.checkpoint.info.metadata["latest_dir"],
            )

        self.run(
            base_model_path=current.model.loaded["hf_model_checkpoint"],
            dataset_path=None,
        )

        current.card.extend(self.config_report())
        self.next(self.end)

    def _get_checkpoint_path(self):
        if not current.checkpoint.is_loaded:
            return None
        print("Using checkpoint from : ", current.checkpoint.info.pathspec)
        if "latest_dir" not in current.checkpoint.info.metadata:
            return None
        print(
            "Using checkpoint from : ", current.checkpoint.info.metadata["latest_dir"]
        )
        print("data in checkpoint directory:", os.listdir(current.checkpoint.directory))
        return os.path.join(
            current.checkpoint.directory,
            current.checkpoint.info.metadata["latest_dir"],
        )

    @step
    def end(self):
        print("Completed!")


if __name__ == "__main__":
    LlamaInstructionTuning()

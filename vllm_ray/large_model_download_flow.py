from metaflow import (
    FlowSpec,
    step,
    huggingface_hub,
    current,
    kubernetes,
    card,
    environment,
    pypi,
    model,
    project,
)
from metaflow import Parameter


class ModelRegistryFlow(FlowSpec):
    """
    A flow to use as a model registry for downloading / accessing models from Hugging Face
    in downstream flows for inference or training.
    """

    # ID of the model from Hugging Face that we will download and use
    model_id = Parameter(
        "model-id",
        help="Hugging Face model ID to download and use",
        default="meta-llama/Llama-3.1-70B-Instruct",
    )

    @step
    def start(self):
        # Move to the next step: downloading the model from Hugging Face
        self.next(self.pull_model_from_huggingface)

    # Alter the resources in this flow based on the need of the model
    @kubernetes(
        cpu=30,
        memory=250 * 1000,  # Set memory requirements in MB
        disk=1500 * 1000,  # Set disk space in MB
        use_tmpfs=True,  # Use in-memory filesystem for faster I/O
        tmpfs_size=150 * 1000,  # Set tmpfs size in MB
    )
    @pypi(
        python="3.11.5",
        packages={
            "huggingface-hub[hf_transfer]": "0.25.2"
        },  # Installing Hugging Face Hub with transfer feature
    )
    # @secrets # Use this decorator to ensure HF_TOKEN is loaded in the environment for authenticated access
    @environment(
        vars={
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable Hugging Face transfer acceleration
        }
    )
    # Hugging Face Hub should download the model into memory for faster upload/download processes.
    # If tmpfs settings are disabled, set `temp_dir_root` to None.
    @huggingface_hub(temp_dir_root="/metaflow_temp/checkpoints")
    @step
    def pull_model_from_huggingface(self):
        import time

        # Record the time taken to download the model
        start_time = time.time()
        self.base_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.*",
            ],  # Download only model weights and tokenizer files
            max_workers=30,  # Use up to 100 threads for parallel download
        )
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.next(self.end)
        # Move to the next step: distributed inference with Ray

    @step
    def end(self):
        # Tag the run with the model ID for future reference
        current.run.add_tags([self.model_id])
        print("Flow has completed")


if __name__ == "__main__":
    ModelRegistryFlow()

from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    secrets,
    model,
    current,
    environment,
    tensorboard,
    pypi,
    huggingface_hub,
    conda,
    card,
    IncludeFile,
    torchrun,
    Parameter,
)
from metaflow.profilers import gpu_profile

LLAMA_FACTORY_GIT_URL = "https://github.com/hiyouga/LLaMA-Factory.git"


class LlamaFactoryMultinodeJob(FlowSpec):

    training_config = IncludeFile(
        "training-config",
        help="JSON file containing training configuration",
        default="train_llama3.2_instruct_lora.json",
    )

    num_nodes = Parameter(
        "num-nodes",
        help="Number of nodes to use for training",
        default=2,
        type=int,
    )

    @step
    def start(self):
        import json

        self.hyperparameters = json.loads(self.training_config)
        self.next(self.pull_model_from_huggingface)

    # Users can comment out the `kubernetes` decorator once the model
    # # has been downloaded for faster local iterations.
    @kubernetes(
        cpu=40,
        memory=200 * 1000,  # Set memory requirements in MB
        disk=1500 * 1000,  # Set disk space in MB
        use_tmpfs=True,  # Use in-memory filesystem for faster I/O
        tmpfs_size=150 * 1000,  # Set tmpfs size in MB
    )
    @huggingface_hub(temp_dir_root="/metaflow_temp/checkpoints")
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
    @step
    def pull_model_from_huggingface(self):
        import time
        import json

        # Record the time taken to download the model
        start_time = time.time()
        self.base_model = current.huggingface_hub.snapshot_download(
            repo_id=self.hyperparameters["model_name_or_path"],
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.*",
            ],  # Download only model weights and tokenizer files
            max_workers=40,  # Use up to 40 threads for parallel download
        )
        end_time = time.time()
        self.time_taken = end_time - start_time
        # Move to the next step: distributed inference with Ray
        self.next(self.tune, num_parallel=self.num_nodes)

    # Llama Factory will require setting HF_TOKEN which can be set via the
    # `@secrets` decorator like below. Customize this as needed.
    # @secrets(sources=['huggingface-token'])
    @torchrun(all_nodes_started_timeout=20 * 60)
    @gpu_profile(interval=3)
    @card
    @environment(
        vars={
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    @model(load=[("base_model", "./base_model")])
    @tensorboard
    @pypi(
        python="3.11",
        packages={
            "GitPython": "3.1.43",
            "tensorboard": "2.17.1",
            "llamafactory[torch,metrics,vllm,liger-kernel,bitsandbytes,hqq,gptq,aqlm,vllm,galore,badam,adam-mini,qwen,modelscope] @ git+https://github.com/hiyouga/LLaMA-Factory.git": "@ea5f3ecd46d6c02f972ef68414408ed7d0d64491",
            "accelerate": "0.34.2",
            "transformers": "4.45.0",
            "deepspeed": "0.15.1",
            "omegaconf": "2.4.0.dev3",
        },
    )
    @kubernetes(
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.13",
        gpu=4,
        memory=600 * 1000,
        cpu=100,
        # Allocate the disk according to the need
        shared_memory=15 * 1000,  # Allocate 40GB of shared memory
        node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",  # Select A100 NVLINK 80GB GPU nodes
    )  # image contains git as a vendored dependency
    @step
    def tune(self):
        import subprocess
        import json
        import sys
        import os
        from git import Repo
        import tempfile
        import shutil

        # We need to clone the LLaMA-Factory repository so that
        # the datasets within the repository can be accessed.
        # IF we don't clone the repo and and call the subprocess command
        # then llama factory will crash.
        local_dir = tempfile.mkdtemp()
        repo = Repo.clone_from(LLAMA_FACTORY_GIT_URL, local_dir)

        # Get a copy of the current environment variables
        # and add the path to the fastbakery binary to the PATH
        # **This is a temporary workaround. This PATH should NOT be hardcoded.**
        # Ideally the image baked by fast bakery should automatically add the
        # fastbakery binary to the PATH.
        env = os.environ.copy()
        env["PATH"] = "/opt/outerbounds/fastbakery/bin:" + env["PATH"]

        # The following configuration is required to train this with multi-node
        # deepspeed training.
        env["FORCE_TORCHRUN"] = "1"
        env["NNODES"] = str(self.num_nodes)
        env["RANK"] = str(current.parallel.node_index)
        env["MASTER_PORT"] = "29500"
        env["MASTER_ADDR"] = current.parallel.main_ip

        self.output_dir = "llama3_lora"

        self.hyperparameters["model_name_or_path"] = os.path.abspath(
            current.model.loaded["base_model"]
        )
        self.hyperparameters["logging_dir"] = self.obtb.log_dir
        self.hyperparameters["report_to"] = "tensorboard"
        self.hyperparameters["output_dir"] = self.output_dir

        json.dump(
            self.hyperparameters,
            open(os.path.join(local_dir, "train_cfg.json"), "w", encoding="utf-8"),
            indent=2,
        )

        proc = subprocess.run(
            ["llamafactory-cli", "train", "train_cfg.json"],
            env=env,
            cwd=local_dir,
        )
        if proc.returncode != 0:
            print(
                f"Training subprocess failed with return code {proc.returncode}. Exiting."
            )
            sys.exit(proc.returncode)
        if current.parallel.node_index == 0:
            self.trained_model_info = current.model.save(
                os.path.join(local_dir, self.output_dir), metadata=self.hyperparameters
            )
            shutil.rmtree(local_dir)
        self.next(self.join)

    @step
    def join(self, inputs):
        for i in inputs:
            if getattr(i, "trained_model_info", None):
                self.trained_model_info = i.trained_model_info
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    LlamaFactoryMultinodeJob()

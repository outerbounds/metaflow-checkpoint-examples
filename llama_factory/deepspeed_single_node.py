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
)
from metaflow.profilers import gpu_profile

LLAMA_FACTORY_GIT_URL = "https://github.com/hiyouga/LLaMA-Factory.git"


class LlamaFactorySingleJob(FlowSpec):

    training_config = IncludeFile(
        "training-config",
        help="JSON file containing training configuration",
        default="train_llama3.2_instruct_lora.json",
    )

    @step
    def start(self):
        self.next(self.tune)

    # Llama Factory will require setting HF_TOKEN which can be set via the
    # `@secrets` decorator like below. Customize this as needed.
    # @secrets(sources=['huggingface-token'])
    @gpu_profile(interval=3)
    @card
    @environment(
        vars={
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    @model
    @tensorboard
    @pypi(
        python="3.11",
        packages={
            "GitPython": "3.1.43",
            "tensorboard": "2.17.1",
            "llamafactory[torch,metrics,vllm,liger-kernel,bitsandbytes,hqq,gptq,aqlm,vllm,galore,badam,adam-mini,qwen,modelscope] @ git+https://github.com/hiyouga/LLaMA-Factory.git": "@3df39f37e1b7988720f36729830889a4c05399b1",
            "accelerate": "0.34.2",
            "transformers": "4.45.0",
            "deepspeed": "0.15.1",
            "omegaconf": "2.4.0.dev3",
        },
    )
    @kubernetes(
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.13",
        gpu=4,
        memory=100 * 1000,
        cpu=40,
        shared_memory=15 * 1000,  # Allocate 40GB of shared memory
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
        env["FORCE_TORCHRUN"] = "1"

        self.output_dir = "llama3_lora"
        self.hyperparameters = json.loads(self.training_config)
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
        self.trained_model_info = current.model.save(
            os.path.join(local_dir, self.output_dir), metadata=self.hyperparameters
        )
        shutil.rmtree(local_dir)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    LlamaFactorySingleJob()

from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    secrets,
    model,
    current,
    environment,
    card,
    IncludeFile,
)


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
    @card
    @model
    @kubernetes(image="public.ecr.aws/outerbounds/llama-factory", gpu=1)
    @step
    def tune(self):
        import subprocess
        import json
        import sys
        import os

        _current_dir = os.getcwd()
        os.chdir("/LLaMA-Factory")
        self.output_dir = "llama3_lora"
        self.hyperparameters = json.loads(self.training_config)
        json.dump(
            self.hyperparameters,
            open("train_cfg.json", "w", encoding="utf-8"),
            indent=2,
        )
        proc = subprocess.run(["llamafactory-cli", "train", "train_cfg.json"])
        if proc.returncode != 0:
            print(
                f"Training subprocess failed with return code {proc.returncode}. Exiting."
            )
            sys.exit(proc.returncode)
        self.trained_model_info = current.model.save(
            self.output_dir, metadata=self.hyperparameters
        )
        os.chdir(_current_dir)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    LlamaFactorySingleJob()

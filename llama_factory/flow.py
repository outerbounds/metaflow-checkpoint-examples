from metaflow import (
    FlowSpec,
    step,
    kubernetes,
    secrets,
    model,
    current,
    environment,
    card,
)


class LlamaFinetuneGridSearch(FlowSpec):

    shared_params = {
        "stage": "sft",
        "do_train": True,
        "dataset": "identity,alpaca_en_demo",
        "template": "llama3",
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": "llama3_lora",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "warmup_ratio": 0.1,
        "save_steps": 1000,
        "learning_rate": 5e-5,
        "num_train_epochs": 1.1,
        "max_samples": 500,
        "max_grad_norm": 1.0,
        "loraplus_lr_ratio": 16.0,
        "fp16": True,
        "use_liger_kernel": True,
    }

    @step
    def start(self):
        self.hp_config = [
            # {"model_name_or_path": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"},
            {"model_name_or_path": "unsloth/Llama-3.2-1B-Instruct"},
        ]
        self.next(self.tune, foreach="hp_config")

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
        self.hyperparameters = self.input
        self.hyperparameters |= self.shared_params
        self.hyperparameters |= {"output_dir": self.output_dir}
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
        self.next(self.join)

    @step
    def join(self, inputs):
        self.model_info = []
        for i in inputs:
            self.model_info.append(i.trained_model_info)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    LlamaFinetuneGridSearch()

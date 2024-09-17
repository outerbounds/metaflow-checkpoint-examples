import os
import json
from metaflow import (
    FlowSpec,
    step,
    IncludeFile,
    Parameter,
    secrets,
    resources,
    secrets,
    retry,
    pypi_base,
    pypi,
    nvidia,
    kubernetes,
    S3,
    model,
    checkpoint,
    current,
    environment,
    huggingface_hub,
)
from metaflow.profilers import gpu_profile
from exceptions import GatedRepoError, GATED_HF_ORGS


@pypi_base(
    python="3.12",
    packages={
        "datasets": "2.21.0",
        "torch": "2.4.1",
        "transformers": "4.44.2",
        "peft": "0.12.0",
        "trl": "0.10.1",
        "accelerate": "0.34.2",
        "bitsandbytes": "0.43.3",
        "sentencepiece": "0.2.0",
        "safetensors": "0.4.5",
    },
)
class FinetuneLlama3LoRA(FlowSpec):

    script_args_file = IncludeFile(
        "script_args",
        help="JSON file containing script arguments",
        default="hf_peft_args.json",
    )

    smoke = Parameter("smoke", type=bool, default=False, help="Flag for a smoke test")

    @step
    def start(self):
        from my_peft_tools import ScriptArguments

        args_dict = json.loads(self.script_args_file)
        self.script_args = ScriptArguments(**args_dict)
        self.next(self.download_model)

    # @kubernetes(cpu=2, memory=10000, )
    @huggingface_hub
    @step
    def download_model(self):
        self.model_reference = current.huggingface_hub.snapshot_download(
            repo_id=self.script_args.model_name,
            allow_patterns=[
                "*.json",
                "*.safetensors",
            ],
        )
        self.next(self.sft)

    @environment(vars={"TOKENIZERS_PARALLELISM": "true"})
    @checkpoint
    @model(load="model_reference")
    @gpu_profile(interval=1)
    @nvidia
    @step
    def sft(self):
        import os
        from my_peft_tools import create_model, create_trainer, save_model
        from hf_trainer_callback import MetaflowCallback

        model, tokenizer = create_model(
            self.script_args, current.model.loaded["model_reference"]
        )
        trainer = create_trainer(
            self.script_args,
            tokenizer,
            model,
            smoke=self.smoke,
            callbacks=[
                MetaflowCallback(),
            ],
        )
        trainer.train()
        output_dirname, merge_output_dirname = save_model(self.script_args, trainer)
        self.model = current.model.save(output_dirname, label="lora")
        if merge_output_dirname:
            self.merged_model = current.model.save(
                merge_output_dirname, label="lora_fused"
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Model saved with metadata {self.model}")


if __name__ == "__main__":
    FinetuneLlama3LoRA()

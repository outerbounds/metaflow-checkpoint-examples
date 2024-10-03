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
        # Calling the `current.huggingface_hub.snapshot_download` function will download
        # the model from huggingface and save it to metaflow' datastore if it is not present
        # in the Metaflow's datastore. If the model is present, it will just
        # return a reference to the model. In any case the output of this function is a
        # reference to the model.
        # This reference can be loaded in other @steps using the `@model` decorator
        # with the `load` parameter.
        self.model_reference = current.huggingface_hub.snapshot_download(
            repo_id=self.script_args.model_name,
            allow_patterns=[
                "*.json",
                "*.safetensors",
            ],
        )
        self.next(self.preprocess_dataset)

    @step
    def preprocess_dataset(self):
        """
        This is an example of the pre-processing function that will pre-process some data
        and transform it into the format which is acceptable during training.

        Users can modify the function as per their requirements and even plugin custom datasets
        that might be coming from outside huggingface.

        The pattern is fine as long as the dataset can fit in memory. If the dataset is too large then
        a better pattern is to save the dataset in the Metaflow's datastore (like S3) and then pull the
        dataset based on the need.
        """
        from datasets import load_dataset

        dataset = load_dataset(
            self.script_args.dataset_name, streaming=True, split="train"
        )
        self.training_dataset = []
        for sample in iter(dataset):
            instruction = str(sample["instruction"])
            input_text = str(sample.get("input", ""))
            out_text = str(sample["output"])
            if not input_text:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{out_text}"
                    f"<|eot_id|><|end_of_text|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
                    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{out_text}"
                    f"<|eot_id|><|end_of_text|>"
                )
            self.training_dataset.append({"text": formatted_prompt})

        self.next(self.sft)

    @environment(
        vars={
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    @checkpoint
    @model(load="model_reference")
    @gpu_profile(interval=1)
    @kubernetes(
        gpu=1,
        cpu=12,
        memory=32000,
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.9",
    )
    @step
    def sft(self):
        import os
        from my_peft_tools import create_model, create_trainer, save_model, push_to_hub
        from hf_trainer_callback import MetaflowCheckpointCallback

        model, tokenizer = create_model(
            self.script_args, current.model.loaded["model_reference"]
        )
        trainer = create_trainer(
            self.script_args,
            tokenizer,
            model,
            # The `self.training_dataset` is created in the previous step
            # after some pre-processing.
            self.training_dataset,
            smoke=self.smoke,
            callbacks=[
                MetaflowCheckpointCallback(),
            ],
        )
        trainer.train()
        output_dirname, merge_output_dirname = save_model(self.script_args, trainer)
        self.model = current.model.save(output_dirname, label="lora")
        if merge_output_dirname:
            # In many cases, users may want models that are fully fused with the LoRA
            # adapter. This makes it such that the model is very different from the
            # checkpoints that are saved during training (Where the checkpoints are LoRA adapters
            # but the model is the original model with the LoRA adapter fused).
            # Using the `current.model.save` allows these models to live as first class models
            # within Metaflow also enabling a means to track thier lineage.
            self.merged_model = current.model.save(
                merge_output_dirname, label="lora_fused"
            )

        if self.script_args.push_to_hub:
            push_to_hub(
                trainer,
                self.script_args.model_name,
                # This is the model it was finetuned on.
            )
        self.next(self.end)

    @step
    def end(self):
        print(f"Model saved with metadata {self.model}")


if __name__ == "__main__":
    FinetuneLlama3LoRA()

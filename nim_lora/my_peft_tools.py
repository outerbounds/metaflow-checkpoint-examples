import os
import json
import tarfile
import tempfile
from typing import Optional
from dataclasses import dataclass, field, asdict
from dataclasses import dataclass, field
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from huggingface_hub import upload_folder
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


torch.manual_seed(42)


def push_to_hub(
    trainer,
    finetuned_from,
    commit_message="End of Training Run",
    blocking=True,
):
    """
    Ideally call this function once you have saved the model and then want to push the model to hub.
    """
    from huggingface_hub import upload_folder
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    if trainer.hub_model_id is None:
        trainer.init_hf_repo()

    _readme_path = os.path.join(trainer.args.output_dir, "README.md")
    if os.path.exists(_readme_path):
        os.remove(_readme_path)

    trainer.create_model_card(
        finetuned_from=finetuned_from,
    )

    return upload_folder(
        repo_id=trainer.hub_model_id,
        folder_path=trainer.args.output_dir,
        commit_message=commit_message,
        run_as_future=not blocking,
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
    )


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    output_dir: str = field(default="./results")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_r: int = field(default=32)
    max_seq_length: int = field(default=512)
    model_name: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    dataset_name: Optional[str] = field(default="tatsu-lab/alpaca")
    use_4bit: bool = field(default=True)
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: str = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    packing: bool = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: str = field(default="cosine")
    max_steps: int = field(default=-1)
    warmup_steps: Optional[int] = field(default=100)
    group_by_length: bool = field(default=True)
    save_steps: Optional[int] = field(default=0)
    logging_steps: int = field(default=25)
    merge: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)
    push_to_hub: bool = field(default=False)

    def __post_init__(self):
        if self.push_to_hub and self.hub_model_id is None:
            raise ValueError("hub_model_id is required for push_to_hub=True")

    def to_dict(self):
        return asdict(self)


def create_model(args, model_path):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = None

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_trainer(args, tokenizer, model, dataset, smoke=False, callbacks=[]):
    training_arguments = TrainingArguments(
        hub_model_id=args.hub_model_id,
        # Where/how to write results?
        output_dir=args.output_dir,
        logging_steps=1 if smoke else args.logging_steps,
        disable_tqdm=True,
        # How long to train?
        max_steps=1 if smoke else args.max_steps,
        num_train_epochs=args.num_train_epochs,
        # How to train?
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        group_by_length=args.group_by_length,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        log_level="debug",
    )
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    def _generator_dataset():
        for sample in dataset:
            yield {"text": sample["text"]}

    train_dataset = Dataset.from_generator(lambda: _generator_dataset())
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        callbacks=callbacks,
    )
    return trainer


def save_model(args, trainer, dirname="final", merge_dirname="final_merged_checkpoint"):
    output_dir = os.path.join(args.output_dir, dirname)
    trainer.save_model(args.output_dir)

    if args.merge:
        """
        This conditional block merges the LoRA adapter with the original model weights.
        NOTE: For use with NIM, we do not need to do the merge.
        """
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = model.merge_and_unload()
        output_merged_dir = os.path.join(args.output_dir, merge_dirname)
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        return output_dir, output_merged_dir
    else:
        return args.output_dir, None


def download_latest_checkpoint(
    lora_dir="./loras",
    # s3_key='lora_adapter.tar.gz',
    flow_name="FinetuneLlama3LoRA",
):
    from metaflow import load_model, Flow

    os.makedirs(lora_dir, exist_ok=True)
    latest_successful_run = Flow(flow_name).latest_successful_run
    load_model(latest_successful_run.data.model, lora_dir)
    print(f"Checkpoint downloaded and extracted to: {lora_dir}")

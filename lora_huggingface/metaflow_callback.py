from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from metaflow import FlowSpec, Task, current
from typing import Any, Dict, Literal, Optional, Union
from metaflow import Checkpoint
import os


def _make_metrics_dict(metric, filter_metrics):
    data = {}
    for k in metric:
        if k in filter_metrics:
            data[k] = round(metric[k], 3)
    return data


class MetaflowCheckpointCallback(TrainerCallback):
    _pointer_id = 0

    @classmethod
    def bump_id(cls):
        cls._pointer_id += 1

    @classmethod
    def default_name(
        cls,
    ):
        return "chckpt_" + str(cls._pointer_id)

    def __init__(
        self,
        name: Optional[str] = None,
        metrics=[
            "loss",
            "accuracy",
            # "learning_rate"
        ],
    ):
        self.bump_id()
        self._name = name or self.default_name()
        from metaflow import current

        self.checkpointer = (
            current.checkpoint if current.is_running_flow else Checkpoint(init_dir=True)
        )
        # TODO : add support for outside metaflow main process.
        self._default_chckpt_dir = self.checkpointer.directory
        self.latest_checkpoint = None
        self._files = []
        self._metrics = metrics

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        if not state.is_world_process_zero:
            return
        _output_dir = args.output_dir
        if self._default_chckpt_dir in args.output_dir:
            _output_dir = _output_dir.replace(self._default_chckpt_dir, "")
            if _output_dir.startswith("/"):
                _output_dir = _output_dir[1:]

        dir_prefix = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        chckpt_path = os.path.join(args.output_dir, dir_prefix)
        self.latest_checkpoint = self.checkpointer.save(
            chckpt_path,
            metadata={
                "global_step": state.global_step,
                "epoch": round(state.epoch or 0, 3),
                "saved_from": "HuggingfaceTrainer",
                "checkpoint_dir": dir_prefix,
                **_make_metrics_dict(state.log_history[-1], self._metrics),
            },
            name=self._name,
            latest=True,
        )

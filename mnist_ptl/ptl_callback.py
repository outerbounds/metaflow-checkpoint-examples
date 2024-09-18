from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from metaflow import current
from typing import Any, Dict, Literal, Optional, Union
import os
from pytorch_lightning import Trainer
from metaflow import Checkpoint


class MetaflowCheckpoint(ModelCheckpoint):
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
        self, *args, name: Optional[str] = None, set_latest: bool = True, **kwargs
    ):
        self.bump_id()
        super().__init__(*args, **kwargs)
        # TODO : add support for outside metaflow main process.
        monitor = kwargs.get("monitor", None)
        if monitor is not None:
            monitor + "-" + str(self._pointer_id)
        self._chckpt_name = name or self.default_name()
        self.checkpointer = Checkpoint()
        self.latest_checkpoint = None
        self.set_latest = set_latest

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        metadata = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "pbar_dict": trainer.progress_bar_metrics,
            "callback_info": {
                "monitor": self.monitor,
                "mode": self.mode,
            },
            "saved_from": "PTLTrainer",
        }
        self._post_save_callback(metadata, filepath)

    def _post_save_callback(self, metadata, filepath):
        metadata["file_names"] = [os.path.basename(filepath)]
        self.latest_checkpoint = self.checkpointer.save(
            # Ideally it will be one single file!
            filepath,
            metadata,
            name=self._chckpt_name,
            latest=self.set_latest,
        )

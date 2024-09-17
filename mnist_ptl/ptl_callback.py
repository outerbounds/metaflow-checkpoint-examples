from datetime import timedelta
from pytorch_lightning.callbacks import ModelCheckpoint
from metaflow import current
from typing import Any, Dict, Literal, Optional, Union
import os
from pytorch_lightning import Trainer
from metaflow.checkpoint_utils import Checkpoint


class MetaflowPTLCheckpoint(ModelCheckpoint):
    _pointer_id = 0

    @classmethod
    def bump_id(cls):
        cls._pointer_id += 1

    @classmethod
    def default_name(
        cls,
    ):
        return "metaflow_ptl_checkpoint_" + str(cls._pointer_id)

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        self.bump_id()
        super().__init__(*args, **kwargs)
        # TODO : add support for outside metaflow main process.
        self._chckpt_name = name or self.default_name()
        self.checkpointer = Checkpoint()
        self.latest_checkpoint = None
        self._current_set = set()

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
        }
        self._post_save_callback(metadata)

    def _post_save_callback(self, metadata):
        new_keys = set(self.best_k_models.keys())
        old_keys = self._current_set
        if len(new_keys - old_keys) == 0:
            return
        # TODO [PRE-RELEASE] Ensure that we only save **one** file per checkpoint since there is only
        # one new file added here
        metadata["file_names"] = [os.path.basename(k) for k in new_keys - old_keys]
        self.latest_checkpoint = self.checkpointer.save(
            list(new_keys), metadata, name=self._chckpt_name
        )
        self._current_set = new_keys

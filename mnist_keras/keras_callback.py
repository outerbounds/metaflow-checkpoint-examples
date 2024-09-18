from keras.callbacks import ModelCheckpoint
from metaflow import FlowSpec, Task, current
from typing import Any, Dict, Literal, Optional, Union
import glob
from metaflow import Checkpoint
import re


def replace_formatting(format_string, replace_with=""):
    # This regex pattern matches content within curly braces, including the braces themselves
    # It looks for any character that is not a closing brace `}`, to avoid matching nested structures
    pattern = r"\{[^}]*\}"
    # Replace the matched formatting expressions with an empty string
    cleaned_string = re.sub(pattern, replace_with, format_string)
    return cleaned_string


class MetaflowKerasCheckpoint(ModelCheckpoint):

    _pointer_id = 0

    @classmethod
    def bump_id(cls):
        cls._pointer_id += 1

    @classmethod
    def default_name(
        cls,
    ):
        return "metaflow_keras_checkpoint_" + str(cls._pointer_id)

    def __init__(self, *args, name: Optional[str] = None, **kwargs):
        self.bump_id()
        super().__init__(*args, **kwargs)
        self._chckpt_name = name or self.default_name()
        self.checkpointer = Checkpoint()
        self.latest_checkpoint = None
        self._files = []
        self.latest_file = None

    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        logs = logs or {}
        search_pattern = replace_formatting(self.filepath, replace_with="*")
        files = glob.glob(search_pattern)
        if set(files) == set(self._files):
            return
        latest_file = self._get_file_path(epoch, batch, logs)
        self.latest_checkpoint = self.checkpointer.save(
            latest_file,
            name=self._chckpt_name,
            metadata={
                "epoch": epoch,
                "latest_file": latest_file,
                "saved_from": "KerasTrainer",
            },
        )
        self._files = files
        self.latest_file = latest_file

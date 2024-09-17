from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from metaflow import current
from metaflow.checkpoint_utils import Checkpoint
from typing import Any, Dict, Literal, Optional, Union
import os

import glob
from pathlib import Path
import shutil
import psutil


def print_disk_info():
    """
    Prints the current size of the disk, current disk usage percentage,
    and the current space left on the disk in GB.
    """
    # Using psutil to get disk usage statistics
    total, used, free = (
        psutil.disk_usage("/").total,
        psutil.disk_usage("/").used,
        psutil.disk_usage("/").free,
    )

    # Convert bytes to gigabytes for a more human-readable format
    total_gb = total / (1024**3)
    used_percentage = (used / total) * 100
    free_gb = free / (1024**3)

    print(f"Disk size: {total_gb:.2f} GB")
    print(f"Disk usage: {used_percentage:.2f}%")
    print(f"Space left: {free_gb:.2f} GB")


def get_root_paths(paths):
    """
    Filters a list of paths to return only the root-level parent paths using pathlib.

    This function accepts a list of strings representing paths, where each path is manipulated
    using the pathlib library for better structure and readability. It returns a list of paths
    that do not have any subpaths within the input list.

    Parameters:
    - paths (list of str): A list of strings where each string represents a path.

    Returns:
    - list of str: A filtered list containing only root-level parent paths.

    Example:
    - Input: ["a/b/c", "a/b", "d/e", "x/y/z"]
    - Output: ["a/b", "d/e", "x/y/z"]

    Note:
    - Paths are considered case-sensitive.
    - The function assumes that the input paths are normalized (no trailing slashes, etc.).
    """
    root_paths = {Path(p) for p in paths}  # Convert each path string to a Path object.
    non_root_paths = set()

    for path in root_paths:
        # Iterate through all the parents of the path and if any parent is found in root_paths, mark the path as non-root.
        for parent in path.parents:
            if parent in root_paths:
                non_root_paths.add(path)
                break

    return [str(path) for path in root_paths - non_root_paths]


class MetaflowCheckpointCallback(TrainerCallback):
    _pointer_id = 0

    @classmethod
    def bump_id(cls):
        cls._pointer_id += 1

    @classmethod
    def default_name(
        cls,
    ):
        return "metaflow_ptl_checkpoint_" + str(cls._pointer_id)

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.bump_id()
        self._name = name or self.default_name()
        if getattr(current, "checkpoint", None):
            self.checkpointer = current.checkpoint
        else:
            self.checkpointer = Checkpoint()

        self.latest_checkpoint = None
        self._files = []
        self.latest_chkpt_dir = None

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if getattr(current, "checkpoint", None):
            self._default_chckpt_dir = current.checkpoint.directory
        else:
            self._default_chckpt_dir = args.output_dir

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.checkpointer is None:
            return
        if not state.is_world_process_zero:
            return

        _output_dir = args.output_dir
        _output_dir_contents = os.listdir(_output_dir)
        _chckdir_name = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        self.latest_chkpt_dir = _chckdir_name
        self.latest_checkpoint = self.checkpointer.save(
            _output_dir,
            # We provide all dirs in the checkpoint because if tar-detects
            # a change in the directory, we will
            metadata={
                "global_step": state.global_step,
                "epoch": state.epoch,
                "training_args": args.to_dict(),
                "latest_dir": self.latest_chkpt_dir,
            },
        )
        print("Saved the checkpoint", self.latest_checkpoint["key"])

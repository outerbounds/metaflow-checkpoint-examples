from operator import is_
from metaflow import (
    FlowSpec,
    current,
    step,
    checkpoint,
    project,
    retry,
    Parameter,
    huggingface_hub,
    pypi,
    model,
)
import tempfile
import hashlib

from metaflow import FlowSpec, current, step, huggingface_hub

HUB_PYPI_PARAMETES = {
    "packages": {"huggingface-hub": "0.16.4", "omegaconf": "2.4.0.dev3"},
    "python": "3.11.5",
}


@project(name="hf_test")
class HFCachedModelFlow(FlowSpec):

    model_name = Parameter(
        "model-name", help="Name of the huggingface model", required=True
    )

    force_download = Parameter(
        "force-download",
        help="Force download the model even if it exists in cache",
        default=False,
        is_flag=True,
    )

    allow_patterns = Parameter(
        "allow-patterns",
        default=None,
        help="Patterns to allow in the model download",
        type=str,
    )

    @pypi(**HUB_PYPI_PARAMETES)
    @huggingface_hub
    @step
    def start(self):
        self.hugging_face_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            force_download=self.force_download,
            allow_patterns="*"
            if self.allow_patterns is None
            else self.allow_patterns.split(","),
        )
        self.next(self.end)

    @model(load=["hugging_face_model"])
    @step
    def end(self):
        print("using checkpoint", self.hugging_face_model["metadata"])
        print(str(self.hugging_face_model))
        print(current.model.loaded["hugging_face_model"])


if __name__ == "__main__":
    HFCachedModelFlow()

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

HUB_PYPI_PARAMETERS = {
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

    @pypi(**HUB_PYPI_PARAMETERS)
    @huggingface_hub
    @step
    def start(self):
        self.model_reference_foo = current.huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            force_download=self.force_download,
            allow_patterns="*"
            if self.allow_patterns is None
            else self.allow_patterns.split(","),
        )

        self.next(self.in_step_load)

    @pypi(**HUB_PYPI_PARAMETERS)
    @huggingface_hub(load=["google-bert/bert-base-uncased"])
    @step
    def in_step_load(self):
        import os

        model_path = current.huggingface_hub.loaded["google-bert/bert-base-uncased"]
        print(
            "Model is loaded in the directory",
            model_path,
            "contents of the directory : ",
            os.listdir(model_path),
        )
        self.next(self.another_step)
    
    @pypi(**HUB_PYPI_PARAMETERS)
    @huggingface_hub(load=[
        {
            "repo_id": "bert-base-uncased",
            "allow_patterns": ["*.json", "tokenizer.txt"],
            "repo_type": "model"
        },
        {
            "repo_id": "t5-small",
            "allow_patterns": ["config.*", "tokenizer.*"],
            "force_download": True # setting `force_download` will ensure that we bust cache 
        }
    ])
    @step
    def another_step(self):
        import os

        model_path = current.huggingface_hub.loaded["t5-small"]
        print(
            "t5-small Model is loaded in the directory",
            model_path,
            "contents of the directory : ",
            os.listdir(model_path),
        )
        
        model_path = current.huggingface_hub.loaded["bert-base-uncased"]
        print(
            "bert-base-uncased Model is loaded in the directory",
            model_path,
            "contents of the directory : ",
            os.listdir(model_path),
        )
        self.next(self.end)


    @model(load=["model_reference_foo"])
    @step
    def end(self):
        import os

        print(
            "Model is loaded in the directory",
            current.model.loaded["model_reference_foo"],
        )
        print(
            "contents of the directory : ",
            os.listdir(current.model.loaded["model_reference_foo"]),
        )


if __name__ == "__main__":
    HFCachedModelFlow()

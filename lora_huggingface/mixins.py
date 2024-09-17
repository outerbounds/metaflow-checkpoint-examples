from metaflow import Parameter, IncludeFile, JSONType
from config import load_config, TrainConfig
from subprocess import check_output
import re
import tempfile

N_GPU = 4
visible_devices = str(list(range(N_GPU)))[1:-1]
HF_IMAGE = "registry.hub.docker.com/valayob/hf-transformer-gpu:4.39.3.1"


def _to_file(file_bytes, extension=None):
    params = {
        "suffix": f".{extension.replace('.', '')}" if extension is not None else None,
        "delete": True,
        "dir": "./",
    }
    latent_temp = tempfile.NamedTemporaryFile(**params)
    latent_temp.write(file_bytes)
    latent_temp.seek(0)
    return latent_temp


class CudaChecks:

    DRIVER_VER = re.compile(b"Driver Version: (.+?) ")
    CUDA_VER = re.compile(b"CUDA Version:(.*) ")

    @classmethod
    def cuda_exists(cls):
        def parse(r, s):
            return r.search(s).group(1).strip().decode("utf-8")

        try:
            out = check_output(["nvidia-smi"])
            return True
        except:
            return False

    @classmethod
    def read_devices(cls):
        out = check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ]
        )
        return [
            dict(
                zip(("name", "device_id", "memory"), (x.strip() for x in l.split(",")))
            )
            for l in out.decode("utf-8").splitlines()
        ]


class ConfigBase:
    """
    Base class for all config needed for this flow as well as any dependent flows.

    This class can be inherited by downstream classes or even used a mixin.

    This class is meant for reuse in Metaflow flows which want to resue the configuration parameters of this training flow so
    that they can call downstream flows with the same configuration parameters.

    Example:
    --------
    - Upstream flow which is preparing data is inheriting the configuration schema / parameters from this class
    - This way correct configuration parsed in both flows while we can also pass the configuration from the upstream flow to the downstream flow while ensuring that the configuration is valid.
    - This pattern is very useful when we have a complex configuration schema and we want to reuse it in multiple flows. These flows may be invoked asynchronously using event handlers, so having a common configuration schema parser is very useful.
    """

    def _resolve_config(self):
        if (
            self.experiment_config is not None
            and self.experiment_config_file is not None
        ):
            raise ValueError("Cannot specify both --config or --config-file")
        elif self.experiment_config is None and self.experiment_config_file is None:
            raise ValueError("Must specify either --config or --config-file")
        if self.experiment_config is not None:
            return load_config(self.experiment_config)
        if self.experiment_config_file is not None:
            temf = _to_file(
                bytes(self.experiment_config_file, "utf-8"),
            )
            return load_config(temf.name)

    _config = None

    @property
    def config(self) -> TrainConfig:
        if self._config is not None:
            return self._config
        self._config = self._resolve_config()
        return self._config

    experiment_config_file = IncludeFile(
        "config-file",
        help="experiment config file path",
        default="experiment_config.yaml",
    )

    experiment_config = Parameter(
        "config", help="experiment config", default=None, type=JSONType
    )

    def config_report(self):
        from metaflow.cards import Markdown
        from omegaconf import OmegaConf

        return [
            Markdown(f"## Experiment Config"),
            Markdown(f"```\n{OmegaConf.to_yaml(self.config)}```"),
        ]


class HuggingFaceLora(ConfigBase):
    def run(
        self,
        base_model_path=None,
        dataset_path=None,
        env=None,
    ):
        import subprocess
        from omegaconf import OmegaConf

        # TODO set `--nproc_per_node` based on `visible_devices` setting.
        visible_devices = self.config.training.visible_devices

        if visible_devices is None or visible_devices == "auto":
            if not CudaChecks.cuda_exists():
                raise ValueError("nvidia-smi not found")
            device_list = CudaChecks.read_devices()
            visible_devices = str(len(device_list))

        if dataset_path is not None:
            self.config.dataset.local_dataset_path = dataset_path
            self.config.dataset.huggingface_dataset_path = None
        if base_model_path is not None:
            self.config.model.base_model = base_model_path
            self.config.model.local_model = True

        config_yml = OmegaConf.to_yaml(self.config)
        tmpfile = _to_file(bytes(config_yml, "utf-8"))
        subprocess.run(
            [
                f"torchrun",
                "--nnodes=1",
                f"--nproc_per_node={visible_devices}",
                f"--master_port={self.config.training.master_port}",
                "tuner.py",
                f"{tmpfile.name}",
            ],
            env=env,
            check=True,
        )

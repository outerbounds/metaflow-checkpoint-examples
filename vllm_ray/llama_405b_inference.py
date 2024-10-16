from metaflow import (
    FlowSpec,
    step,
    huggingface_hub,
    current,
    kubernetes,
    card,
    environment,
    pypi,
    metaflow_ray,
    model,
    project,
)
from metaflow.profilers import gpu_profile
from metaflow.cards import Table, Artifact, Markdown


class Llama405bVLLMFlow(FlowSpec):

    model_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"

    @step
    def start(self):
        self.next(self.pull_model_from_huggingface)

    # Users can comment out the `kubernetes` decorator once the model
    # has been downloaded once for faster local iterations.
    @kubernetes(
        cpu=100,
        memory=1500 * 1000,
        disk=1500 * 1000,
        use_tmpfs=True,
        tmpfs_size=1000 * 1000,
    )  # Setting tmpfs allows having an in-memory filesystem for faster I/O
    @pypi(
        python="3.11.5",
        packages={"huggingface-hub[hf_transfer]": "0.25.2"},
    )
    # @secrets # Use the decorator to ensure HF_TOKEN is loaded in the environment
    @environment(
        vars={
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    # Huggingface hub should ideally download any model into memory so that the entire upload/download process is faster.
    # If the `tmpfs` related settings are disabled then set `temp_dir_root` to None.
    @huggingface_hub(temp_dir_root="/metaflow_temp/checkpoints")
    @step
    def pull_model_from_huggingface(self):
        import time

        # Since the
        start_time = time.time()
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
            max_workers=50,
        )
        end_time = time.time()
        self.time_taken = end_time - start_time

        # Since VLLM Requires Ray under the hood for distributed
        # inference, we can use the `@metaflow_ray` decorator to
        # create elastic ray clusters for distributed offline inference
        self.next(self.ray_inference, num_parallel=2)

    @metaflow_ray(
        all_nodes_started_timeout=20 * 60
    )  # Timeout after 20 minutes if all nodes are not started
    @gpu_profile(interval=0.2)
    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.3",
            "huggingface-hub": "0.25.1",
            "setuptools": "74.1.2",
        },
    )
    @kubernetes(
        cpu=50,
        memory=1000 * 1000,
        disk=1500 * 1000,
        # Llama 405B 8bit will need 500GB of tmpfs
        use_tmpfs=True,
        tmpfs_size=500 * 1000,
        gpu=8,
        shared_memory=40 * 1000,
        # The below node selector is for selecting Coreweave's A100 machines
        node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",
    )
    @environment(
        vars={
            # We need to set VLLM_WORKER_MULTIPROC_METHOD based on the following issue
            # https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )  # The `@model` decorator will the load the model from s3 into a tmpfs path (which makes loading a lot faster)
    # Since the `self.llama_model` contained a reference to the model in Metaflow's datastore, `@model` will load the model
    # onto a path provided in the `load` argument.
    @model(load=[("llama_model", "/metaflow_temp/llama_models")])
    @step
    def ray_inference(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(
            "loading the model from the path",
            current.model.loaded["llama_model"],
        )
        #
        tokenizer = AutoTokenizer.from_pretrained(current.model.loaded["llama_model"])
        # we set enforce_eager so that we don't waste time in the cuda graph calculation.
        llm = LLM(
            model=current.model.loaded["llama_model"],
            tensor_parallel_size=8,
            enforce_eager=False,
            worker_use_ray=True,
            gpu_memory_utilization=0.90,
            max_model_len=1024 + 128,
            # gpu_memory_utilization=0.2,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "user", "content": "How many sailors are on your ship?"},
        ]

        sampling_params = SamplingParams(temperature=0.5)
        outputs = llm.generate(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ),
            sampling_params,
        )

        self.text = outputs[0].outputs[0].text.strip()

        print("done with generation")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @step
    def end(self):
        current.run.add_tags([self.model_id])
        print("ending the flow")


if __name__ == "__main__":
    Llama405bVLLMFlow()

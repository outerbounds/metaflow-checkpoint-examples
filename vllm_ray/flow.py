from metaflow import (
    FlowSpec,
    step,
    card,
    pypi,
    kubernetes,
    parallel,
    secrets,
    huggingface_hub,
    current,
    model,
    metaflow_ray,
)

DISK_SIZE = 100 * 1000  # 100 GB

MEMORY = 60 * 1000  # 60 GB


class VLLMInferenceFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.pull_model_from_huggingface)

    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.1",
            "transformers": "4.44.2",
            "huggingface-hub": "0.25.1",
        },
    )
    @huggingface_hub
    @step
    def pull_model_from_huggingface(self):
        # `current.huggingface_hub.snapshot_download` downloads the model from the Hugging Face Hub
        # and saves it in the backend storage based on the model's `repo_id`. If there exists a model
        # with the same `repo_id` in the backend storage, it will not download the model again. The return
        # value of the function is a reference to the model in the backend storage.
        # This reference can be used to load the model in the subsequent steps via `@model(load=["llama_model"])`

        self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer.*"],
        )
        self.next(self.run_vllm, num_parallel=2)

    # Run Distributed inference with VLLM on Ray requires the model
    # to be loaded on the same path for all worker nodes. We achieve this by
    # providing the `load` argument a list of tuples that contain the reference
    # to the model and the path where the model needs to be unpacked.
    @model(load=[("llama_model", "./llama_model")])
    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.1",
            "transformers": "4.44.2",
            "huggingface-hub": "0.25.1",
            "setuptools": "74.1.2",
        },
    )
    @kubernetes(
        cpu=16,
        gpu=8,
        memory=MEMORY,
        # Enable this node selector to run this flow on selected machines on kubernetes.
        # The below selector will run it on Coreweave's A100 machines.
        # Disable this node selector to run this flow on any machine with requested resources
        # on kubernetes.
        node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",
        # Change the `image` to some other image if you are running with `--environment=pypi` flag.
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.9",
        shared_memory=12 * 1000,  # 12 GB shared memory as ray requires this.
    )
    @metaflow_ray(
        all_nodes_started_timeout=20 * 60
    )  # 20 minute timeout so that all workers start.
    @card
    @step
    def run_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        import huggingface_hub
        import os

        print(
            "loading the model from the path",
            current.model.loaded["llama_model"],
        )
        tokenizer = AutoTokenizer.from_pretrained(current.model.loaded["llama_model"])
        # we set enforce_eager so that we don't waste time in the cuda graph calculation.
        llm = LLM(
            model=current.model.loaded["llama_model"],
            tensor_parallel_size=8,
            enforce_eager=True,
        )

        print("running the model")
        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
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
        self.next(self.end)

    @step
    def end(self):
        print("ending the flow")


if __name__ == "__main__":
    VLLMInferenceFlow()

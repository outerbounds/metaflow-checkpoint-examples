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
    environment,
    model,
    metaflow_ray,
    Parameter,
)

DISK_SIZE = 100 * 1000  # 100 GB

MEMORY = 60 * 1000  # 60 GB


class VLLMInferenceFlow(FlowSpec):

    model_id = Parameter(
        "model-id",
        help="The model id used in the ModelRegistryFlow.",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    @step
    def start(self):
        from metaflow import Flow, Run

        run_list = list(
            Flow("ModelRegistryFlow", _namespace_check=False).runs(self.model_id)
        )
        if len(run_list) == 0:
            raise ValueError(
                f"Model with id {self.model_id} not found in the registry. Please run ModelRegistryFlow to download the model and make it accessible to all executions."
            )
        run = run_list[0]
        self.llama_model = run.data.base_model
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
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.13",
        shared_memory=12 * 1000,  # 12 GB shared memory as ray requires this.
    )
    @metaflow_ray(
        all_nodes_started_timeout=20 * 60
    )  # 20 minute timeout so that all workers start.
    @environment(
        vars={
            # Required to avoid issues with multiprocessing in VLLM
            # See: https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
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

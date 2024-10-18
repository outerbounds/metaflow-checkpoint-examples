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

    # ID of the model from Hugging Face that we will download and use
    model_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"

    @step
    def start(self):
        # Move to the next step: downloading the model from Hugging Face
        self.next(self.pull_model_from_huggingface)

    # Users can comment out the `kubernetes` decorator once the model
    # has been downloaded for faster local iterations.
    @kubernetes(
        cpu=100,
        memory=1500 * 1000,  # Set memory requirements in MB
        disk=1500 * 1000,  # Set disk space in MB
        use_tmpfs=True,  # Use in-memory filesystem for faster I/O
        tmpfs_size=1000 * 1000,  # Set tmpfs size in MB
    )
    @pypi(
        python="3.11.5",
        packages={
            "huggingface-hub[hf_transfer]": "0.25.2"
        },  # Installing Hugging Face Hub with transfer feature
    )
    # @secrets # Use this decorator to ensure HF_TOKEN is loaded in the environment for authenticated access
    @environment(
        vars={
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable Hugging Face transfer acceleration
        }
    )
    # Hugging Face Hub should download the model into memory for faster upload/download processes.
    # If tmpfs settings are disabled, set `temp_dir_root` to None.
    @huggingface_hub(temp_dir_root="/metaflow_temp/checkpoints")
    @step
    def pull_model_from_huggingface(self):
        import time

        # Record the time taken to download the model
        start_time = time.time()
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_id,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer.*",
            ],  # Download only model weights and tokenizer files
            max_workers=50,  # Use up to 50 threads for parallel download
        )
        end_time = time.time()
        self.time_taken = end_time - start_time

        # Move to the next step: distributed inference with Ray
        self.next(self.ray_inference, num_parallel=2)

    @metaflow_ray(
        all_nodes_started_timeout=20
        * 60  # Timeout after 20 minutes if nodes are not started
    )
    @gpu_profile(interval=0.2)  # Profile GPU usage with 0.2 second intervals
    @pypi(
        python="3.10.11",
        packages={
            "vllm": "0.6.3",  # VLLM package for large language model inference
            "huggingface-hub": "0.25.1",  # Hugging Face Hub library
            "setuptools": "74.1.2",  # Python package installer
        },
    )
    @kubernetes(
        cpu=50,
        memory=1000 * 1000,  # Set memory to 1TB
        disk=1500 * 1000,  # Set disk space to 1.5TB
        gpu=8,  # Allocate 8 GPUs for the task
        shared_memory=40 * 1000,  # Allocate 40GB of shared memory
        node_selector="gpu.nvidia.com/class=A100_NVLINK_80GB",  # Select A100 NVLINK 80GB GPU nodes
        image="registry.hub.docker.com/valayob/gpu-base-image:0.0.13",
    )
    @environment(
        vars={
            # Required to avoid issues with multiprocessing in VLLM
            # See: https://github.com/vllm-project/vllm/issues/6152#issuecomment-2211709345
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    # The `@model` decorator loads the model from S3 into tmpfs for faster loading.
    # Since the `self.llama_model` contains a reference to the model in Metaflow's datastore,
    # `@model` will load the model onto a path provided in the `load` argument.
    # The model must be loaded on the same path across all workers for distributed inference.
    @model(load=[("llama_model", "/metaflow_temp/llama_models")])
    @step
    def ray_inference(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # Load the tokenizer and model for inference
        print(
            "Loading the model from the path:",
            current.model.loaded["llama_model"],
        )
        tokenizer = AutoTokenizer.from_pretrained(current.model.loaded["llama_model"])

        # Initialize the LLM model for distributed inference using Ray
        llm = LLM(
            model=current.model.loaded["llama_model"],
            tensor_parallel_size=8,  # Use 8 GPUs for parallel execution
            enforce_eager=False,  # Skip CUDA graph calculation for faster inference
            worker_use_ray=True,  # Use Ray for distributed worker management
            gpu_memory_utilization=0.90,  # Utilize 90% of available GPU memory
            max_model_len=1024 + 128,  # Set maximum token length for input and output
        )

        # Define chat-style interaction prompts for the model
        messages = [
            {
                "role": "system",
                "content": "You are a pirate chatbot who always responds in pirate speak!",
            },
            {"role": "user", "content": "Who are you?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "user", "content": "How many sailors are on your ship?"},
        ]

        # Set sampling parameters for text generation
        sampling_params = SamplingParams(temperature=0.5)

        # Generate text response using the model
        outputs = llm.generate(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ),
            sampling_params,
        )

        # Store the generated text
        self.text = outputs[0].outputs[0].text.strip()

        print("Generation complete")
        self.next(self.join)

    @step
    def join(self, inputs):
        # Merge the results from parallel steps
        self.merge_artifacts(inputs)
        self.next(self.end)

    @step
    def end(self):
        # Tag the run with the model ID for future reference
        current.run.add_tags([self.model_id])
        print("Flow has completed")


if __name__ == "__main__":
    Llama405bVLLMFlow()

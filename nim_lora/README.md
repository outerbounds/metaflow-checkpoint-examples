# Training a Custom LoRA Adapter with Metaflow on @nvidia 

## Setup

1. This example require execution on the Outerbounds Platform. To run it on the outerbounds platform first run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow


### Smoke test
```bash
python finetune_hf_peft.py --environment=pypi run --smoke True
```

### Full run
```bash
python finetune_hf_peft.py --environment=pypi run
```

### Download your lora adapter and move it to `$NIM_PEFT_SOURCE`

```python
import os
from my_peft_tools import download_latest_checkpoint
download_latest_checkpoint(
    lora_dir=os.path.join(os.path.expanduser('~'), 'loras') 
    # NOTE: this is the default
)
```

### Serve NIM container

#### Set up environment
```
export HF_TOKEN=...
export NGC_API_KEY=...
```

```bash
export LOCAL_PEFT_DIRECTORY=$HOME/loras
export NIM_PEFT_SOURCE=$HOME/loras
export NIM_PEFT_REFRESH_INTERVAL=600 
export CONTAINER_NAME=meta-llama3-8b-instruct
export NIM_CACHE_PATH=$HOME/nim-cache
mkdir -p "$NIM_CACHE_PATH"
chmod -R 777 $NIM_CACHE_PATH
```

#### Run container in foreground
```bash
docker run -it --rm --name=$CONTAINER_NAME --runtime=nvidia --gpus all --shm-size=16GB -e NGC_API_KEY=$NGC_API_KEY -e NIM_PEFT_SOURCE -e NIM_PEFT_REFRESH_INTERVAL -v $NIM_CACHE_PATH:/opt/nim/.cache -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE  -p 8000:8000 nvcr.io/nim/meta/llama3-8b-instruct:latest
```

#### Run container in background
```bash
docker run -d -it --rm --name=$CONTAINER_NAME --runtime=nvidia --gpus all --shm-size=16GB -e NGC_API_KEY=$NGC_API_KEY -e NIM_PEFT_SOURCE -e NIM_PEFT_REFRESH_INTERVAL -v $NIM_CACHE_PATH:/opt/nim/.cache -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE  -p 8000:8000 nvcr.io/nim/meta/llama3-8b-instruct:latest
```

#### Query custom LoRA adapter

```bash
curl -X 'POST'   'http://0.0.0.0:8000/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
    "model": "llama3-8b-instruct-alpaca-custom",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      },
      {
        "role":"user",
        "content":"Can you write me a song?"
      }
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
}'
```

## Salient Features

- **Using Custom Models From External Sources like Huggingface**: This flow leverages the `@huggingface_hub` decorator to save and cache models so that they can be used in subsequent steps. More information about this decorator can be found in [this example](../hf_registry/)

- **Checkpointing in external compute mediums**: The `@checkpoint` functionality works in tandem will all compute exection mediums supported by Metaflow such as `@batch`, `@kubernetes` or even `@nvidia`. This checkpointing functionality in this flow piggyback a Huggingface callback mechanism via the `MetaflowCheckpointCallback` mentioned in [the other examples](../lora_huggingface/). 

- **Saving First Class Models For Inference**: The `@model` decorator is used to save the model and provide it first class identity in the Metaflow ecosystem. This maybe needed when the final models maybe different from checkpoints. For example, some models may require fusing the LoRA adapter with the base model so that it can be later used. The below code snippet from the flow showcases the API in action: 
  ```python
  self.model = current.model.save(output_dirname, label="lora")
  if merge_output_dirname:
    self.merged_model = current.model.save(
        merge_output_dirname, label="lora_fused"
    )
  ```
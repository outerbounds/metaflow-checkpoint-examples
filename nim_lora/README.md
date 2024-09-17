### Finetune workflow

#### Smoke test
```bash
python finetune_hf_peft.py --environment=pypi run --smoke True
```

#### Full run
```bash
python finetune_hf_peft.py --environment=pypi run
```

#### Download your lora adapter and move it to `$NIM_PEFT_SOURCE`

```python
import os
from my_peft_tools import download_latest_checkpoint
download_latest_checkpoint(
    lora_name="llama3-8b-instruct-alpaca-custom",  # remember this name, you'll need it later
    lora_dir=os.path.join(os.path.expanduser('~'), 'loras') # NOTE: this is the default
)
```

### Serve container

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
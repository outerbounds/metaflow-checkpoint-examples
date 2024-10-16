# VLLM Inference with Ray

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow On Outerbounds Platform

### Small Models (7B Parameters)
```bash
python flow.py --environment=fast-bakery run 
```


### Large Models (405B Parameters)
```bash
python llama_405b_inference.py --environment=fast-bakery run 
```

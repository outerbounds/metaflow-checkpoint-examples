## Part 1: Installation and WebUI startup
[Read more](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation).

If you are on an Outerbounds workstation, we recommend doing this in the base environment.
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
cp -r data ..
pip install -e ".[torch,metrics]"
cd ..
pip install bitsandbytes liger-kernel
```

If you are _not_ on an Outerbounds workstation, also run:
```bash
pip install outerbounds
```
followed by your configure command visible in the Outerbounds UI.

Start the llama factory web UI

```bash
GRADIO_SHARE=1 llamafactory-cli webui
```

then, start an Outerbounds app
```bash
GRADIO_SERVER_PORT=6000 llamafactory-cli webui
outerbounds app start --port 6000 --name llamafactorywebui
```

after ~2 minutes you'll be able to visit the app URL.

## Part 2: Tune an LLM in Llama Factory using the CLI

NOTE: https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#use-wb-logger

Run the LLaMA Factory trainer
```bash
llamafactory-cli train train_llama3.2_instruct_lora.json 
```

```bash
GRADIO_SHARE=1 llamafactory-cli webchat infer_llama3.2_instruct_lora.json
```

## Part 3: Run Llama Factory Remotely. 

```bash
python single_train_flow.py run --training-config train_llama3.2_instruct_lora.json
```

## Part 4: Running Llama Factory Remotely with FastBakery
    
```bash
python fb_single_train.py --environment=fast-bakery run --training-config train_llama3.2_instruct_lora.json
```

## Part 5: Run grid search in Metaflow code
Let's run a grid search over quantization and model variations.

```bash
python flow.py run
```

## Part 6: Running Llama Factory Remotely with FastBakery and Deepspeed (Single Node Multi-GPU)

```bash
python deepspeed_single_node.py --environment=fast-bakery run --training-config train_llama3.1_instruct_deepspeed.json
```


## Part 7: Running Llama Factory Remotely with FastBakery and Deepspeed (Multi Node Multi-GPU)
    
Ensure that `metaflow-torchrun` is installed in your environment. You can `pip install metaflow-torchrun` to install it.
```bash
python deepspeed_multi_node.py --environment=fast-bakery run --training-config train_llama3.1_instruct_deepspeed.json
```
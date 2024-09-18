# Metaflow `@checkpoint`/`@model`/`@huggingface_hub` Examples

Long-running data processing and machine learning jobs often present several challenges:

1. **Failure Recovery**: Recovering from failures can be painful and time-consuming.
   - *Example*: Suppose you're training a deep learning model that takes 12 hours to complete. If the process crashes at the 10-hour mark due to a transient error, without checkpoints, you'd have to restart the entire training from scratch.
   - *Example*: During data preprocessing, you generate intermediate datasets like tokenized text or transformed images. Losing these intermediates means re-running expensive computations, which can be especially problematic if they took hours to create.

2. **External Dependencies**: Jobs may require large external data (e.g., pre-trained models) that are cumbersome to manage.
   - *Example*: Loading a pre-trained transformer model from Hugging Face Hub can take a significant amount of time and bandwidth. If this model isn't cached, every run or worker node (in a distributed training context) would need to download it separately, leading to inefficiencies.

3. **Version Control in Multi-User Environments**: Managing checkpoints and models in a multi-user setting requires proper version control to prevent overwriting and ensure correct loading during failure recovery.
   - *Example*: If multiple data scientists are training models and saving checkpoints to a shared storage, one user's checkpoint might accidentally overwrite another's. This can lead to confusion and loss of valuable work. Moreover, when a job resumes after a failure, it must load the correct checkpoint corresponding to that specific run and user.

To address these challenges, Metaflow introduces the `@checkpoint`/ `@model`/ `@huggingface_hub` decorators, which simplify the process of saving and loading checkpoints and models within your flows. These decorators ensure that your long-running jobs can be resumed seamlessly after a failure, manage external dependencies efficiently, and maintain proper version control in collaborative environments.

This repository contains a gallery of examples demonstrating how to leverage `@checkpoint`/`@model`/`@huggingface_hub` to overcome the aforementioned challenges. By exploring these examples, you'll learn practical ways to integrate checkpointing and model management into your workflows, enhancing robustness, efficiency, and collaboration.**

---

## Starter Examples

**Basic Checkpointing with `@checkpoint`:**

- [MNIST Training with Vanilla PyTorch](./mnist_torch_vanilla)
- [MNIST Training with Keras](./mnist_keras)
- [MNIST Training with PyTorch Lightning](./mnist_ptl)
- [MNIST Training with Hugging Face Transformers](./mnist_huggingface)
- [Saving XGBoost Models as Part of the Model Registry](./xgboost/)

These starter examples introduce the fundamentals of checkpointing and model saving. They show how to implement `@checkpoint` in simple training workflows, ensuring that you can recover from failures without losing progress. You'll also see how `@model` helps in saving and loading models/checkpoints effortlessly.

---

## Intermediate Examples

**Checkpointing with Large Models and Managing External Dependencies:**

- [Training LoRA Models with Hugging Face](./lora_huggingface/)
- [Training LoRA Models on NVIDIA GPU Cloud with `@nvidia`](./nim_lora/)
- [Generating Videos from Text Using Stable Diffusion XL and Stable Diffusion Video](./stable-diff/)

These intermediate examples dive into more complex scenarios where managing large external models becomes crucial. You'll learn how to use `@checkpoint`/`@model` alongside external resources like Hugging Face Hub (with `@huggingface_hub`). 

---

## Advanced Examples

**Checkpointing and Failure Recovery in Distributed Training Environments:**

- [Multi-node Distributed Training of CIFAR-10 with PyTorch DDP](./cifar_distributed/)

The advanced examples focus on distributed training environments where the complexity of failure recovery and model management increases. You'll explore how `@checkpoint` facilitates seamless recovery across multiple nodes. 
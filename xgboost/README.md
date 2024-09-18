# XGBoost Example

This example, trains a simple XGBoost model and then loads the model for inference in subsequent steps. This example showcases the usage of the `@model` decorator to save and load the model

## Setup

1. If you are running OSS Metaflow install `pip install metaflow`; If you are runnning it on the Outerbounds Platform, run `pip install outerbounds`

2. Install other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Flow
```bash
python flow.py --environment=fast-bakery run
```

## Salient Features
- **Model Storage/Loading**: This flow uses the `@model` decorator to save and load the model. The `@model` decorator exposes a `current.model.save` function that can be used to save models created as a part of the `@step`. These models are all given unique identity and stored as a part of the `@step`'s metadata information. The saved model is loaded in another @step using the `@model(load="xgboost_model")`
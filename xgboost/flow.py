from metaflow import FlowSpec, step, pypi_base, model, current


@pypi_base(
    python="3.12",
    packages={
        "xgboost": "2.1.1",
        "numpy": "2.1.1",
    },
)
class XGBoostModelRegistry(FlowSpec):
    @model
    @step
    def start(self):
        import numpy as np
        import xgboost as xgb

        np.random.seed(0)
        X = np.random.rand(100, 10)
        y = np.random.randint(2, size=100)
        dtrain = xgb.DMatrix(X, label=y)
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        bst.save_model("model.bst")
        self.xgboost_model = current.model.save(
            "model.bst",
            metadata={
                "my_message": "hi",
            },
        )
        self.next(self.end)

    @model(load="xgboost_model")
    @step
    def end(self):
        # current.model.loaded["xgboost_model"] is the path to directory
        # holding the loaded model
        import os
        import xgboost as xgb

        model_path = os.path.join(current.model.loaded["xgboost_model"], "model.bst")
        bst = xgb.Booster()
        bst.load_model(model_path)
        print(
            bst.predict(
                xgb.DMatrix([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
            )
        )


if __name__ == "__main__":
    XGBoostModelRegistry()

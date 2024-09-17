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
        current.model.save(
            "model.bst",
            metadata={
                "my_message": "hi",
            },
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    XGBoostModelRegistry()

import time
from metaflow import FlowSpec, step, Parameter, retry, current
from metrics_logger import metrics_logger, MetricsLogger


class LoggerFlow(FlowSpec):
    @step
    def start(self):
        self.data = [1, 2, 3, 4, 5]
        self.next(self.train, foreach="data")

    @metrics_logger
    @step
    def train(self):
        setattr(self, "data_" + str(self.input), self.input)
        logger = MetricsLogger(
            {
                "flow_name": current.flow_name,
                "run_id": current.run_id,
                "step_name": current.step_name,
                "task_id": current.task_id,
            }
        )

        for i in range(20 * self.input):
            logger.log_step(i, "accuracy", i * 0.01)
            logger.log_step(i, "loss", 1 - i * 0.01)
            logger.log_step(i, "learning_rate", 0.01)
            logger.save()
            time.sleep(0.25)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.end)

    @step
    def end(self):
        print(
            "values of data_1, data_2, data_3, data_4, data_5",
            self.data_1,
            self.data_2,
            self.data_3,
            self.data_4,
            self.data_5,
        )


if __name__ == "__main__":
    LoggerFlow()

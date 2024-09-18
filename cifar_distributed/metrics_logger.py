import json
import os
import tempfile
from typing import Dict
from functools import wraps
from metaflow.cards import Markdown, VegaChart, Table, Artifact
from metaflow import card, current
from threading import Thread, Event
from metaflow.flowspec import INTERNAL_ARTIFACTS_SET
from collections import defaultdict
from metaflow.plugins.cards.card_modules.components import (
    UserComponent,
    with_default_component_id,
)
import time


def update_spec_data(spec, data):
    spec["data"]["values"] = data
    return spec


def update_data_object(data_object, data):
    data_object["values"].append(data)
    return data_object


def line_chart_spec(
    title=None,
    x_name="u",
    y_name="v",
    xtitle=None,
    ytitle=None,
    width=600,
    height=400,
    with_params=True,
    x_axis_temporal=False,
):
    parameters = [
        {
            "name": "interpolate",
            "value": "linear",
            "bind": {
                "input": "select",
                "options": [
                    "basis",
                    "cardinal",
                    "catmull-rom",
                    "linear",
                    "monotone",
                    "natural",
                    "step",
                    "step-after",
                    "step-before",
                ],
            },
        },
        {
            "name": "tension",
            "value": 0,
            "bind": {"input": "range", "min": 0, "max": 1, "step": 0.05},
        },
        {
            "name": "strokeWidth",
            "value": 2,
            "bind": {"input": "range", "min": 0, "max": 10, "step": 0.5},
        },
        {
            "name": "strokeCap",
            "value": "butt",
            "bind": {"input": "select", "options": ["butt", "round", "square"]},
        },
        {
            "name": "strokeDash",
            "value": [1, 0],
            "bind": {
                "input": "select",
                "options": [[1, 0], [8, 8], [8, 4], [4, 4], [4, 2], [2, 1], [1, 1]],
            },
        },
    ]
    parameter_marks = {
        "interpolate": {"expr": "interpolate"},
        "tension": {"expr": "tension"},
        "strokeWidth": {"expr": "strokeWidth"},
        "strokeDash": {"expr": "strokeDash"},
        "strokeCap": {"expr": "strokeCap"},
    }
    spec = {
        "title": title if title else "Line Chart",
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        # "width": width,
        # "height": height,
        "params": parameters if with_params else [],
        "data": {"name": "values", "values": []},
        "mark": {
            "type": "line",
            "tooltip": True,
            **(parameter_marks if with_params else {}),
        },
        "selection": {"grid": {"type": "interval", "bind": "scales"}},
        "encoding": {
            "x": {
                "field": x_name,
                "title": xtitle if xtitle else x_name,
                **({"timeUnit": "seconds"} if x_axis_temporal else {}),
                **({"type": "quantitative"} if not x_axis_temporal else {}),
            },
            "y": {
                "field": y_name,
                "type": "quantitative",
                "title": ytitle if ytitle else y_name,
            },
        },
        **({"width": width, "height": height} if width and height else {}),
    }
    data = {"values": []}
    return spec, data


class LineChart(UserComponent):
    REALTIME_UPDATABLE = True

    def __init__(
        self,
        title,
        xtitle,
        ytitle,
        x_name,
        y_name,
        with_params=False,
        x_axis_temporal=False,
        width=None,
        height=None,
    ):
        super().__init__()

        self.spec, _ = line_chart_spec(
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            x_name=x_name,
            y_name=y_name,
            with_params=with_params,
            x_axis_temporal=x_axis_temporal,
            width=width,
            height=height,
        )

    def update(self, data):  # Can take a diff
        self.spec = update_spec_data(self.spec, data)

    @with_default_component_id
    def render(self):
        vega_chart = VegaChart(self.spec, show_controls=True)
        vega_chart.component_id = self.component_id
        return vega_chart.render()


class MetricsLogger:

    DEFAULT_FILE_NAME = "./metaflow_training_metrics.json"

    def __init__(self, training_args: Dict, save_file_name=None, save_steps=20) -> None:
        self._training_args = training_args
        self._step_metrics = defaultdict(list)
        self._epoch_metrics = defaultdict(list)
        if save_file_name:
            self._save_file_name = save_file_name
        elif "METAFLOW_METRICS_LOGGER_FILE" in os.environ:
            self._save_file_name = os.environ["METAFLOW_METRICS_LOGGER_FILE"]
        else:
            print(
                "[@metrics_logger] No file name provided, using default file name %s"
                % self.DEFAULT_FILE_NAME
            )
            self._save_file_name = self.DEFAULT_FILE_NAME

        self._save_steps = save_steps

    def log_step(self, step: int, dimension: str, value: float) -> None:
        self._step_metrics[dimension].append({"index": step, "value": value})
        if step % self._save_steps == 0:
            self.save()

    def log_epoch(self, epoch: int, dimension: str, value: float) -> None:
        self._epoch_metrics[dimension].append({"index": epoch, "value": value})
        self.save()

    def save(self):
        with open(self._save_file_name, "w") as f:
            json.dump(
                {
                    "training_args": self._training_args,
                    "step_metrics": self._step_metrics,
                    "epoch_metrics": self._epoch_metrics,
                },
                f,
                indent=4,
            )


class MetricsCardRendererThread(Thread):

    CARD_ID = "training_metrics_card"

    def __init__(self, metrics_file: str, interval=1) -> None:
        super().__init__()
        self._file_name = metrics_file
        self._interval = interval
        self._exit_event = Event()
        self.daemon = True
        self.card_refresher = MetricsCardRefresher()
        self._current_card = current.card[self.CARD_ID]

    def _safely_load(self):
        try:
            with open(self._file_name, "r") as f:
                return json.load(f), None
        except FileNotFoundError as e:
            return {}, e
        except json.JSONDecodeError as e:
            return {}, e
        except Exception as e:
            return {}, e

    def run(self):
        self.card_refresher.on_startup(self._current_card)
        while self._exit_event.is_set() is False:
            data, current_error = self._safely_load()
            if current_error:
                self.card_refresher.on_error(self._current_card, current_error)
            else:
                # print("Got some data", data)
                self.card_refresher.on_update(self._current_card, data)
            time.sleep(self._interval)

    def stop(self):
        self._exit_event.set()
        self.join()


def json_to_table(json_data):
    data = []
    for k in json_data:
        data.append([Markdown("**%s**" % k), Artifact(json_data[k])])
    return Table(data=data)


class MetricsCardRefresher:
    def __init__(self):
        self._rendered = False
        self.current = current
        self._step_charts = {}
        self._epoch_charts = {}

    def _header_components(self):
        return [
            Markdown("# Training Metrics"),
            Markdown(
                "**Task %s [Attempt:%s]**"
                % (self.current.pathspec, self.current.retry_count)
            ),
        ]

    def _training_args_components(self, data_object):
        return (
            []
            if len(data_object["training_args"]) == 0
            else [
                Markdown("## Training Args"),
                json_to_table(data_object["training_args"]),
            ]
        )

    def _footer_components(self):
        return []

    def _refreshable_components(self, data_object):
        step_metrics = data_object[
            "step_metrics"
        ]  # {dimension: [{index: int, value: float}]}
        epoch_metrics = data_object[
            "epoch_metrics"
        ]  # {dimension: [{index: int, value: float}]}
        components = [
            Markdown("## Metrics"),
        ]
        for dimension in step_metrics:
            if dimension not in self._step_charts:
                self._step_charts[dimension] = LineChart(
                    title="Step Metrics [%s]" % dimension,
                    xtitle="Step",
                    ytitle=dimension,
                    x_name="index",
                    y_name="value",
                    x_axis_temporal=False,
                    width=600,
                    height=400,
                )
            self._step_charts[dimension].update(step_metrics[dimension])
            components.append(self._step_charts[dimension])
        for dimension in epoch_metrics:
            if dimension not in self._epoch_charts:
                self._epoch_charts[dimension] = LineChart(
                    title="Epoch Metrics[%s]" % dimension,
                    xtitle="Epoch",
                    ytitle=dimension,
                    x_name="index",
                    y_name="value",
                    x_axis_temporal=False,
                    width=600,
                    height=400,
                )
            self._epoch_charts[dimension].update(epoch_metrics[dimension])
            components.append(self._epoch_charts[dimension])

        return components

    def _update_refreshable_components(self, current_card, data_object):
        epoch_metrics = data_object["epoch_metrics"]
        step_metrics = data_object["step_metrics"]
        for dimension in epoch_metrics:
            self._epoch_charts[dimension].update(epoch_metrics[dimension])
        for dimension in step_metrics:
            self._step_charts[dimension].update(step_metrics[dimension])
        current_card.refresh()

    def on_startup(self, current_card):
        current_card.extend(self._header_components())
        current_card.extend(self._footer_components())
        current_card.refresh()

    def first_time_render(self, current_card, data_object):
        current_card.clear()
        current_card.extend(self._header_components())
        current_card.extend(self._training_args_components(data_object))
        current_card.extend(self._refreshable_components(data_object))
        current_card.extend(self._footer_components())
        current_card.refresh()

    def on_error(self, current_card, error_message):
        pass

    def _check_for_new_dimensions(self, data_object):
        step_metrics = data_object["step_metrics"]
        epoch_metrics = data_object["epoch_metrics"]
        for dimension in step_metrics:
            if dimension not in self._step_charts:
                return True
        for dimension in epoch_metrics:
            if dimension not in self._epoch_charts:
                return True
        return False

    def on_update(self, current_card, data_object):
        # print("Recieved Data Object", data_object)
        if not set(["training_args", "step_metrics", "epoch_metrics"]).issubset(
            set(data_object.keys())
        ):
            return

        if not self._rendered or self._check_for_new_dimensions(data_object):
            self.first_time_render(current_card, data_object)
            self._rendered = True
        else:
            self._update_refreshable_components(current_card, data_object)


def metrics_logger(func):
    # This is a step decorator that will start a thread and then read the metrics file
    # and then render a card based on the metrics file.
    INTERNAL_ARTIFACTS_SET.add("_metrics_info")

    @wraps(func)
    def wrapper(self):
        try:
            from metaflow import current

            with tempfile.NamedTemporaryFile() as f:
                os.environ["METAFLOW_METRICS_LOGGER_FILE"] = f.name
                thread = MetricsCardRendererThread(f.name)
                thread.start()
                func(self)
                self._metrics_info, _ = thread._safely_load()
        finally:
            thread.stop()

    return card(type="blank", id=MetricsCardRendererThread.CARD_ID, customize=False)(
        wrapper
    )

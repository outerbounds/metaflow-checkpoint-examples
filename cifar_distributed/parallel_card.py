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


def make_gang_layout(
    current_node_index,
    num_nodes,
):
    data_values = []
    for i in range(num_nodes):
        if i == current_node_index:
            _type = "self"
            if i == 0:
                _type = "self (control)"
            data_values.append({"index": i, "type": _type})
        elif i == 0:
            data_values.append({"index": i, "type": "control"})
        else:
            data_values.append({"index": i, "type": "worker"})

    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 600,
        "height": 100,
        "padding": 5,
        "data": [{"name": "table", "values": data_values}],
        "scales": [
            {
                "name": "xscale",
                "type": "band",
                "domain": {"data": "table", "field": "index"},
                "range": "width",
            },
            {
                "name": "color",
                "type": "ordinal",
                "domain": {"data": "table", "field": "type"},
                "range": ["#FF6347", "#4682B4", "#32CD32"],
            },
        ],
        "axes": [
            {
                "orient": "bottom",
                "scale": "xscale",
                "title": "Gang Cluster Layout",
                "labelColor": "black",
                "labelFontSize": 14,
                "tickColor": "black",
            }
        ],
        "legends": [
            {
                "fill": "color",
                "title": "Type",
                "orient": "right",
                "encode": {
                    "symbols": {
                        "enter": {
                            "fillOpacity": {"value": 1},
                            "stroke": {"value": "transparent"},
                        }
                    }
                },
            }
        ],
        "marks": [
            {
                "type": "rect",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"scale": "xscale", "field": "index"},
                        "width": {"scale": "xscale", "band": 1, "offset": -2},
                        "y": {"value": 10},
                        "height": {"value": 80},
                        "fill": {"scale": "color", "field": "type"},
                    }
                },
            }
        ],
    }


class ParallelCardRendererThread(Thread):

    CARD_ID = "parallel_card"

    def __init__(self, interval=1) -> None:
        super().__init__()
        self._interval = interval
        self._exit_event = Event()
        self.daemon = True
        self.card_refresher = ParallelCardRefresher()
        self._current_card = current.card[self.CARD_ID]

    def _safely_load(self):
        return {}, None

    def run(self):
        self.card_refresher.on_startup(self._current_card)
        while self._exit_event.is_set() is False:
            data, current_error = self._safely_load()
            if current_error:
                self.card_refresher.on_error(self._current_card, current_error)
            else:
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


class ParallelCardRefresher:
    def __init__(self):
        self._rendered = False
        self.current = current
        self._step_charts = {}
        self._epoch_charts = {}

    def _header_components(self):
        return [
            Markdown(
                "# Gang Scheduled Cluster [%s]"
                % (
                    "control-task"
                    if self.current.parallel.node_index == 0
                    else "worker-task"
                )
            ),
            Markdown(
                "**Task %s [Attempt:%s]**"
                % (self.current.pathspec, self.current.retry_count)
            ),
            Markdown("## Cluster Info"),
            Table(
                data=[
                    [
                        Markdown("**Host Name**"),
                        Artifact(get_my_ip()),
                    ],
                    [
                        Markdown("**Cluster Size**"),
                        Artifact(self.current.parallel.num_nodes),
                    ],
                    [
                        Markdown("**Current Node Index**"),
                        Artifact(self.current.parallel.node_index),
                    ],
                ]
            ),
            Markdown("## Cluster Layout"),
            VegaChart(
                make_gang_layout(
                    self.current.parallel.node_index, self.current.parallel.num_nodes
                ),
            ),
        ]

    def _footer_components(self):
        return []

    def _refreshable_components(self, data_object):
        return []

    def _update_refreshable_components(self, current_card, data_object):
        pass

    def on_startup(self, current_card):
        current_card.extend(self._header_components())
        current_card.extend(self._footer_components())
        current_card.refresh()

    def first_time_render(self, current_card, data_object):
        current_card.clear()
        current_card.extend(self._header_components())
        current_card.extend(self._refreshable_components(data_object))
        current_card.extend(self._footer_components())
        current_card.refresh()

    def on_error(self, current_card, error_message):
        pass

    def on_update(self, current_card, data_object):
        if not self._rendered:
            self.first_time_render(current_card, data_object)
            self._rendered = True
        else:
            self._update_refreshable_components(current_card, data_object)


def get_my_ip():
    import socket

    return socket.gethostbyname(socket.gethostname())


def parallel_card(func):
    @wraps(func)
    def wrapper(self):
        try:
            from metaflow import current

            thread = ParallelCardRendererThread()
            thread.start()
            func(self)

        finally:
            thread.stop()

    return card(type="blank", id=ParallelCardRendererThread.CARD_ID, customize=False)(
        wrapper
    )

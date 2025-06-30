#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Contains functions for wrapper logs
"""
# pylint: disable=too-many-lines

import logging
import os
import time
import uuid
from multiprocessing import Pipe
from threading import Thread
from json_checker import Checker

import pandas as pd  # type: ignore

try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    import psutil  # type: ignore

    _HAS_PLOTLY_PSUTIL = True
except ImportError:
    _HAS_PLOTLY_PSUTIL = False


class Profiler:
    """
    Main profiler class for Pandora
    """

    enabled = False
    save_graphs = False
    save_raw_data = False
    _profiling_info = pd.DataFrame(columns=["level", "parent", "name", "uuid", "time", "call_time", "memory"])
    running_processes = []

    @staticmethod
    def enable_from_config(conf: dict):
        """
        Enables the profiler if specified in the config file

        :param conf: The configuration dict
        :type conf: dict
        """

        if _HAS_PLOTLY_PSUTIL:

            base_conf = conf.get("profiling", False)

            if isinstance(base_conf, bool):
                base_conf = {
                    "save_graphs": base_conf,
                    "save_raw_data": base_conf,
                }

            elif isinstance(base_conf, dict):
                base_conf.update(
                    {
                        "save_graphs": base_conf.get("save_graphs", False),
                        "save_raw_data": base_conf.get("save_raw_data", False),
                    }
                )

            else:
                raise TypeError("The 'profiling' key in the configuration has to be either a dict or a boolean.")

            schema = {
                "save_graphs": bool,
                "save_raw_data": bool,
            }

            checker = Checker(schema)
            checker.validate(base_conf)

            Profiler.save_graphs = base_conf["save_graphs"]
            Profiler.save_raw_data = base_conf["save_raw_data"]
            Profiler.enabled = Profiler.save_graphs or Profiler.save_raw_data
        else:
            pass

    @staticmethod
    def add_profiling_info(info: dict):
        """
        Add profiling info to the profiling DataFrame.

        :param info: dictionary with profiling data keys
        :type info: dict
        """

        Profiler._profiling_info.loc[len(Profiler._profiling_info)] = {
            "level": info["level"],
            "parent": info["parent"],
            "name": info["name"],
            "uuid": info["uuid"],
            "time": info["time"],
            "call_time": info["call_time"],
            "memory": info["memory"],
        }

    @staticmethod
    def generate_summary(base_output: str):
        """
        Generate Profiling summary

        :param base_output: Pandora's output directory
        :type base_output: str
        """

        if not Profiler.enabled:
            return

        output = os.path.join(base_output, "profiling")

        if Profiler.save_raw_data or Profiler.save_graphs:
            os.makedirs(output, exist_ok=True)

        if Profiler.save_raw_data:
            Profiler._profiling_info.to_pickle(os.path.join(output, "raw_data.pickle"))

        Profiler._profiling_info["text_display"] = (
            Profiler._profiling_info["name"] + " (" + Profiler._profiling_info["time"].round(2).astype(str) + " s)"
        )

        if Profiler.save_graphs:
            # time profiling flame graph
            fig = px.icicle(
                Profiler._profiling_info,
                names="text_display",
                ids="uuid",
                parents="parent",
                values="time",
                title="Time profiling icicle graph (functions tagged only)",
                color="time",
                color_continuous_scale="thermal",
                branchvalues="total",
            )

            fig.update_traces(tiling_orientation="v")

            fig.write_html(os.path.join(output, "time_graph.html"))

            # memory profiling graph
            for _, call_row in Profiler._profiling_info[Profiler._profiling_info["memory"].notnull()].iterrows():
                fig = Profiler.plot_trace_for_call(call_row["uuid"], "memory")

                if fig:
                    fig.write_html(os.path.join(output, "memory_{}.html".format(call_row["name"])))

    @staticmethod
    def plot_trace_for_call(call_uuid, data_name):
        """
        Plot memory (or any resource tracked) usage over time for a function call, with markers for its subcalls.

        :param call_uuid: UUID of the parent function call
        :type call_uuid: str
        :param data_name: The name of the data to plot (if cpu consumption were to be added for example)
        :type data_name: str

        :return: The generated plotly figure
        :rtype: plotly.graph_objs._figure.Figure
        """

        # Get the parent call entry
        parent_row = Profiler._profiling_info[Profiler._profiling_info["uuid"] == call_uuid]
        if parent_row.empty:
            return None

        parent_row = parent_row.iloc[0]

        call_start_time = parent_row["call_time"]
        times = [data[0] - call_start_time for data in parent_row[data_name]]
        values = [data[1] for data in parent_row[data_name]]

        # Collect subcalls (direct children)
        subcalls = Profiler._profiling_info[Profiler._profiling_info["parent"] == call_uuid]

        # Plot memory usage line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=values, mode="lines+markers", name=f"{data_name} usage"))

        # Base Y position for markers
        base_y = max(values)
        offset_step = (max(values) - min(values)) / 50  # how much higher each subsequent label goes
        current_offset = -offset_step * 2

        for _, row in list(subcalls.iterrows())[::-1]:
            sub_t = row["call_time"] - call_start_time
            sub_name = row["name"]

            y_position = base_y + current_offset

            fig.add_trace(
                go.Scatter(
                    x=[sub_t],
                    y=[y_position],
                    mode="markers+text",
                    marker={
                        "color": "black",
                        "size": 8,
                    },
                    text=[sub_name],
                    textposition="middle right",  # text right next to marker at same height
                    showlegend=False,
                )
            )

            fig.add_shape(
                type="line",
                x0=sub_t,
                x1=sub_t,
                y0=min(values),
                y1=y_position,
                line={
                    "color": "black",
                    "width": 1,
                    "dash": "dot",
                },
            )

            # Increment offset for next marker
            current_offset += offset_step

        fig.update_layout(
            title="{} usage during {} call".format(data_name, parent_row["name"]),
            xaxis_title="Time (s)",
            yaxis_title="Memory (MB)",
            showlegend=True,
        )

        return fig


def profile(name, interval=0.05, memprof=False):
    """
    Pandora profiling decorator

    :param name: name of the function in the report
    :type name: str
    :param interval: memory sampling interval (seconds)
    :type interval: int or float
    :param memprof: whether to profile the memory consumption
    :type memprof: bool
    """

    def decorator_generator(func):
        """
        Inner function
        """

        def wrapper_profile(*args, **kwargs):
            """
            Profiling wrapper

            Generate profiling logs of function, run

            :return: func(*args, **kwargs)
            """

            # if profiling is disabled, remove overhead
            if not Profiler.enabled:
                return func(*args, **kwargs)

            call_uuid = str(uuid.uuid4())
            parent_uuid = Profiler.running_processes[-1] if Profiler.running_processes else "__main__"
            level = len(Profiler.running_processes)

            func_name = name
            if name is None:
                func_name = func.__name__.capitalize()

            Profiler.running_processes.append(call_uuid)

            if memprof:
                # Launch memory profiling thread
                child_pipe, parent_pipe = Pipe()
                thread_monitoring = MemProf(os.getpid(), child_pipe, interval=interval)
                thread_monitoring.start()
                if parent_pipe.poll(1):  # wait for thread to start
                    parent_pipe.recv()

            start_time = time.time()
            res = func(*args, **kwargs)
            total_time = time.time() - start_time

            if memprof:
                # end memprofiling monitoring
                parent_pipe.send(0)

            Profiler.running_processes.pop(-1)  # remove function from call list

            func_data = {
                "level": level,
                "parent": parent_uuid,
                "name": func_name,
                "uuid": call_uuid,
                "time": total_time,
                "call_time": start_time,
                "memory": thread_monitoring.mem_data if memprof else None,
            }

            Profiler.add_profiling_info(func_data)

            return res

        return wrapper_profile

    return decorator_generator


class MemProf(Thread):
    """
    MemProf

    Profiling thread
    """

    def __init__(self, pid, pipe, interval):
        """
        Init function of MemProf

        :param pid: The process ID of the monitored process
        :type pid: int
        :param pipe: The pipe used to send the end monitoring signal
        :type pipe: multiprocessing.connection.Connection
        :param interval: Time interval (seconds) between memory measurements
        :type interval: float
        """
        super().__init__()
        self.pipe = pipe
        self.interval = interval
        self.process = psutil.Process(pid)
        self.mem_data = []

    def run(self):
        """
        Run the memory profiling thread
        """

        try:
            # tell parent profiling is ready
            self.pipe.send(0)

            while True:

                timestamp = time.time()

                # Get memory in megabytes
                current_mem = self.process.memory_info().rss / 1000000
                self.mem_data.append((timestamp, current_mem))

                if self.pipe.poll(self.interval):
                    break

        except BrokenPipeError:
            logging.debug("Broken pipe error in log wrapper.")

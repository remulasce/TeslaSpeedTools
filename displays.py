import string

import pandas
import plotly.graph_objs
from plotly import graph_objects as go

import data
from data import C


def make_subplot_graph(fig: plotly.graph_objs.Figure, trace_data: pandas.DataFrame, x_axis: string = "R torque",
                       y_axis: string = "Battery current", row: int = 1, col: int = 1,
                       torque_estimate=None, color: string = None, name: string = None,
                       trace_kwargs: dict = {"secondary_y": False}, scatter_kwargs: dict = {}) -> None:
    if name is None:
        name = str(y_axis)
    else:
        name = str(name)
    local_data = trace_data.sort_values(by=x_axis, inplace=False)

    marker = {
        'size': 5,
        'color': data.colors_map[y_axis] if color is None else color}
    fig.add_trace(
        go.Scattergl(
            x=local_data[x_axis],
            y=local_data[y_axis],
            mode="markers",
            marker=marker,
            name=name,
            **scatter_kwargs
        ),
        **trace_kwargs,
        row=row, col=col,
    )
    fig.update_yaxes(title_text=y_axis, row=row, col=col, secondary_y=trace_kwargs["secondary_y"])
    fig.update_xaxes(title_text=x_axis, row=row, col=col)

    if torque_estimate is not None:
        fig.add_trace(
            expected_power_from_throttle(local_data, x_axis, torque_estimate), row=row, col=col
        )


def expected_power_from_throttle(local_data, axis,
                                 estimate_fun=lambda frame: -59.24 + (3.59 * frame[C.ACCELERATOR_PEDAL])):
    trend_x = local_data[axis]
    trend_y = local_data.apply(estimate_fun, axis=1)

    return go.Scattergl(
        x=trend_x, y=trend_y, name="prediction", mode="markers", marker={'color': 'red'}
    )


def show_figure(fig):
    fig.update_layout(hoverlabel_namelength=-1)
    # fig.update_xaxes(spikemode="across+marker", spikethickness=2)
    # fig.update_traces(marker={'color': 'black'})
    fig.show_dash(config=dict({'scrollZoom': True}))
    # fig.show(config=dict({'scrollZoom': True}))

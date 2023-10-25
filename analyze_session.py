import sys

import numpy
from plotly import subplots
from plotly_resampler import register_plotly_resampler

import displays
import predict_torque
from data import read_files, C
from displays import make_subplot_graph


def main():
    file = sys.argv[1]
    print(f"Analyzing session: {file}")
    register_plotly_resampler(mode='auto', default_n_shown_samples=750)

    fig = analyze_temperatures(read_analysis_file(file), file)
    displays.show_figure(fig)


def analyze_temperatures(trace_data, filename):
    rows = 6
    cols = 1
    row_heights = [200, 300, 200, 150, 150, 200]
    fig = subplots.make_subplots(rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=False,
                                 row_heights=row_heights,
                                 horizontal_spacing=0.1, vertical_spacing=0.025,
                                 column_titles=[str(filename)],
                                 specs=numpy.full((rows, cols), {"secondary_y": True}).tolist())

    i = 1
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.SPEED, row=i, col=1)
    fig.update_yaxes(row=i, col=1, autorange=False, range=[0, 200])

    i += 1
    i = plot_temperatures(fig, i, trace_data)

    i += 1
    plot_flows(fig, i, trace_data)

    i += 1
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.PERCENT_TORQUE_CUT, row=i, col=1)
    fig.update_yaxes(row=i, col=1, autorange=False, range=[0, 25])

    i = i + 1
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_TORQUE, row=i, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_POWER, row=i, col=1,
                       trace_kwargs={"secondary_y": "true"})
    fig.update_yaxes(secondary_y=False, row=i, col=1, range=[0, 400])
    fig.update_yaxes(secondary_y=True, row=i, col=1, range=[0, 250])

    print("Displaying")

    fig.update_layout(dragmode="pan", overwrite=True)
    fig.update_yaxes(fixedrange=True, autorange=False)
    fig.update_layout(height=sum(row_heights))
    fig.update_xaxes(autorange=False, range=[0, 6 * 60],
                     tickmode='linear', dtick=60, tick0=0, showticklabels=True,
                     title_text=None
                     )
    return fig


def read_analysis_file(file):
    print("Reading file")
    trace_data = read_files(file, interpolate=True)
    print("Adding torque predictions")
    predict_torque.add_torque_prediction_trace(trace_data)
    trace_data = trace_data.sample(frac=.01)
    return trace_data


def plot_temperatures(fig, row, trace_data):
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP, row=row, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.FRONT_OIL_TEMP, row=row, col=1,
                       trace_kwargs={'secondary_y': True})
    fig.update_yaxes(row=row, col=1, autorange=False, range=[30, 120])
    fig.update_yaxes(row=row, col=1, autorange=False, range=[20, 80], secondary_y=True)

    row += 1
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.CELL_TEMP_MID, row=row, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.BATTERY_INLET, row=row, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.POWERTRAIN_INLET, row=row, col=1)
    fig.update_yaxes(row=row, col=1, autorange=False, range=[20, 60], title_text="Coolant Temps")

    return row


def plot_flows(fig, row, trace_data):
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.FRONT_OIL_FLOW, row=row, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.BATTERY_FLOW, row=row, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.POWERTRAIN_FLOW, row=row, col=1)
    fig.update_yaxes(row=row, col=1, autorange=False, range=[0, 30])


if __name__ == '__main__':
    main()

import plotly.express as px
import plotly.subplots as subplots
from plotly_resampler import register_plotly_resampler, FigureResampler

import models
import predict_torque
from displays import make_subplot_graph, show_figure
from models import curve_fit_torque
from predict_torque import add_torque_prediction_trace
from data import C, filter_pedal_application, read_files, Files, TorquePredictionFiles

register_plotly_resampler(mode='auto')


def main():
    model_files = TorquePredictionFiles.all

    # fig = review_trace(Files.east_1)
    # show_figure(fig)
    # return

    # model = models.SRPLUS_TORQUE_MODEL_PARAMS

    print("Plotting....")
    display_files = [TorquePredictionFiles.no_oil_cooler_fastlap]
    # fig = plot_stator_cooling(files=[Files.bw_200_fastlap_trace])
    # fig = compare_traces(display_files)
    fig = review_single_trace(Files.bw_200_fastlap_trace)
    show_figure(fig)
    print("Done")


def compare_traces(files):
    # trace_data = read_files(files)
    fig = subplots.make_subplots(rows=len(files), cols=2, shared_xaxes=True, shared_yaxes=True)
    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.SPEED, row=i, col=1)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.R_TORQUE, row=i, col=1,
                           torque_estimate=modelled_torque_estimate())

        make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.R_STATOR_TEMP, row=i, col=2)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.BATTERY_INLET, row=i, col=2)

    fig.update_layout(height=len(files * 600))
    return fig


def review_single_trace(file):
    # trace_data = read_files(files)
    n_rows = 5
    fig = subplots.make_subplots(rows=n_rows, cols=1, vertical_spacing=.05, shared_xaxes=True, shared_yaxes=True)

    trace_data = read_files(file)
    add_torque_prediction_trace(trace_data)
    trace_data = trace_data.sample(frac=.01)

    make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.SPEED, row=1, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.R_TORQUE, row=1, col=1)
    make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.PREDICTED_MAX_TORQUE, row=1, col=1)

    throttled_traces = trace_data[trace_data[C.PERCENT_TORQUE_CUT] > 0]
    stator_deltas = predict_torque.calculate_stator_estimates(throttled_traces)

    make_subplot_graph(fig, throttled_traces, x_axis=C.TIME, y_axis=C.R_STATOR_TEMP, row=2, col=1)
    make_subplot_graph(fig, stator_deltas, x_axis=C.TIME, y_axis=C.R_STATOR_TEMP_D, row=2, col=1)
    make_subplot_graph(fig, throttled_traces, x_axis=C.TIME, y_axis=C.PERCENT_TORQUE_CUT, row=2, col=1)

    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.TARGET_BAT_ACTIVECOOL, row=3, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.TARGET_BAT_PASSIVECOOL, row=3, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.BATTERY_INLET, row=3, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.PERCENT_TORQUE_CUT, row=3, col=1)
    #
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.TARGET_PT_ACTIVECOOL, row=4, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.TARGET_PT_PASSIVECOOL, row=4, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.POWERTRAIN_INLET, row=4, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.PERCENT_TORQUE_CUT, row=4, col=1)
    #
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.BATTERY_FLOW, row=5, col=1)
    # make_subplot_graph(fig, trace_data, x_axis=C.TIME, y_axis=C.POWERTRAIN_FLOW, row=5, col=1)

    fig.update_layout(height=n_rows * 600)
    return fig


def modelled_torque_estimate():
    return lambda frame: models.predict_torque_frame(
        frame,
        **models.SRPLUS_TORQUE_MODEL_PARAMS
    )


def plot_torque_cut_v_temp(files):
    cols = 1
    titles = [str(file) for file in files for _ in range(cols)]
    fig = subplots.make_subplots(rows=len(files) * 3, cols=cols, shared_xaxes=True, shared_yaxes=True,
                                 subplot_titles=titles)

    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file)

        trace_data = filter_pedal_application(trace_data, pedal_min=75)

        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP, row=i, col=1, name=file)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.FRONT_OIL_TEMP, row=i, col=1)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.PERCENT_TORQUE_CUT, row=i, col=1, name=file)
        i = i + 1
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis="Battery inlet", row=i, col=1)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis="Powertrain inlet", row=i, col=1)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis="Cell temp mid", row=i, col=1)
        i = i + 1
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis="Powertrain flow", row=i, col=1)
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis="Battery flow", row=i, col=1)

    fig.add_hline(y=110)
    fig.update_layout(height=len(files * 600 * 3))
    return fig

if __name__ == '__main__':
    main()

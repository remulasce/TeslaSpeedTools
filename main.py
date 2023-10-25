import plotly.express as px
import plotly.subplots as subplots
from plotly_resampler import register_plotly_resampler, FigureResampler

import models
import predict_torque
from displays import make_subplot_graph, show_figure
from predict_torque import add_torque_prediction_trace
from data import C, filter_pedal_application, read_files, Files, TorquePredictionFiles
from stator_cooling import plot_stator_cooling

register_plotly_resampler(mode='auto')


def main():
    print("Analyzing....")
    # model_files = [
    #     Files.bw_23_1_8_S4_unthrottled,
    #     Files.west_fast_unthrottled, Files.east_unthrottled, Files.th_1, Files.th_3, Files.th_5, Files.th_6]
    model_files = TorquePredictionFiles.all

    # fig = review_trace(Files.east_1)
    # show_figure(fig)
    # return

    # model = models.SRPLUS_TORQUE_MODEL_PARAMS
    # model = tune_all_params(files=files)
    model = tune_fw_log_constants(files=model_files)
    # model = tune_params(files=Files.th_all)

    print("Plotting....")
    display_files = [Files.east_2_throttled]
    # fig = plot_speed_vs_torque_and_power(model, files=display_files)
    fig = plot_stator_cooling(files=[Files.bw_200_fastlap_trace])
    # fig = compare_traces(display_files)
    # fig = review_single_trace(Files.bw_200_fastlap_trace)
    show_figure(fig)
    print("Done")


def tune_all_params(files=Files.th_all):
    trace_data = read_files(files)
    trace_data = filter_pedal_application(trace_data, pedal_min=75, pedal_max=100)
    trace_data = models.filter_over_cliff(trace_data)

    return models.make_predictions(trace_data)


def tune_fw_log_constants(files):
    return models.tune_voltage_logarithmic_constants(files)


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


def plot_speed_vs_torque_and_power(model=None, files=Files.th_all):
    titles = [str(file) for file in files for _ in range(2)]
    fig = subplots.make_subplots(rows=len(files), cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file)

        trace_data = filter_pedal_application(trace_data, pedal_min=75)

        # make_subplot_graph(fig, trace_data, x_axis=C.ACCELERATOR_PEDAL, y_axis="R torque", row=i, col=1, name=file)
        # make_subplot_graph(fig, trace_data, x_axis=C.ACCELERATOR_PEDAL, y_axis="R torque", row=i, col=1, name=file,
        #                    torque_estimate=modelled_torque_estimate())

        make_subplot_graph(fig, trace_data, x_axis=C.SPEED, y_axis=C.ACCELERATOR_PEDAL, row=i, col=1, color='gray')
        make_subplot_graph(fig, trace_data, x_axis="Speed", y_axis="R torque", row=i, col=1,
                           torque_estimate=modelled_torque_estimate())

    fig.update_layout(height=len(files * 600))
    return fig


def modelled_torque_estimate():
    return lambda frame: models.predict_torque_onerow(
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


def make_px_graph(cols, trace_data):
    fig = px.line(trace_data, x="Time Merged",
                  y=cols,
                  range_x=[0, 50000], range_y=[0, 500])  # range doesn't work.
    fig = FigureResampler(
        default_n_shown_samples=100000,
        figure=fig
    )
    fig.update_xaxes(
        rangeslider=dict(
            visible=True)

    )
    fig.layout.xaxis.update(range=[50, 50000])
    fig.update_xaxes(spikemode="across+marker", spikethickness=2)
    fig.show(config=dict({'scrollZoom': True}))


if __name__ == '__main__':
    main()

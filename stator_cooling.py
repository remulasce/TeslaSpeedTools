import math
from math import isnan

import numpy
import pandas as pd
from numpy import NaN
from pandas import DataFrame
from plotly import subplots as subplots
from plotly_resampler import register_plotly_resampler
from statsmodels.nonparametric.smoothers_lowess import lowess

import displays
import predict_torque
from displays import make_subplot_graph

from data import Files, read_files, C
from test_shit import test_shit


def main():
    print("Analyzing stator cooling")
    register_plotly_resampler(mode='auto')

    plot_stator_cooling([Files.cal_precondition])
    # plot_stator_cooling([Files.bw_200_fastlap_trace])


def plot_stator_cooling(files):
    subplot_cols = 1
    subplot_rows = len(files) * 3

    # titles = [str(file) for file in files for _ in range(subplot_cols)]
    fig = subplots.make_subplots(rows=subplot_rows, cols=subplot_cols, shared_xaxes=True, shared_yaxes=False,
                                 horizontal_spacing=0.1, vertical_spacing=0.01,
                                 # subplot_titles=titles,
                                 specs=numpy.full((subplot_rows, subplot_cols), {"secondary_y": True}).tolist()
                                 )

    fig.update_layout(height=len(files * 600 * 3))

    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file, interpolate=True)

        predict_torque.add_torque_prediction_trace(trace_data)
        trace_data = trace_data.sample(frac=.01)

        trace_data = add_smooth_stator_temp(trace_data)
        trace_data = add_stator_d(trace_data, C.R_STATOR_TEMP, C.R_STATOR_TEMP_D)
        # trace_data = add_stator_d(trace_data, C.R_STATOR_TEMP_SMOOTH, C.R_STATOR_TEMP_SMOOTH_D)
        trace_data = add_smooth_stator_temp(trace_data, C.R_STATOR_TEMP_D, C.R_STATOR_TEMP_SMOOTH_D)

        # Only show where torque was cut.
        trace_data = trace_data[trace_data[C.PERCENT_TORQUE_CUT] > 0]

        # Analyze the data
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP, row=i, col=1)
        # make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_SMOOTH, row=i, col=1)

        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.PERCENT_TORQUE_CUT, row=i, col=1)
        fig.add_hline(row=i, y=110)
        fig.add_hline(row=i, y=100)
        # make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_SMOOTH_D, row=i, col=1,
        #                    trace_kwargs={'secondary_y': True})
        # make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_D, row=i, col=1,
        #                    trace_kwargs={'secondary_y': True})
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_SMOOTH_D, row=i, col=1,
                           trace_kwargs={'secondary_y': True})
        fig.update_yaxes(secondary_y=True, row=i, col=1, range=[0, 2])

        i = i + 1
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_SMOOTH_D, row=i, col=1,
                           trace_kwargs={'secondary_y': False})
        make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.R_STATOR_TEMP_D, row=i, col=1,
                           trace_kwargs={'secondary_y': False})
        fig.update_yaxes(secondary_y=False, row=i, col=1, range=[0, 10])
        # make_subplot_graph(fig, trace_data, x_axis=C.TIME_MERGED, y_axis=C.FRONT_OIL_TEMP, row=i, col=1)
        i = i + 1

    displays.show_figure(fig)


def add_smooth_stator_temp(trace_data, input_column=C.R_STATOR_TEMP, output_column=C.R_STATOR_TEMP_SMOOTH):
    stator_data = trace_data[[C.TIME_MERGED, input_column]]

    # Select our own x-values to smooth to for consistency
    xvals = numpy.arange(stator_data[C.TIME_MERGED].min(), stator_data[C.TIME_MERGED].max(), .1)
    lowess_estimate = lowess(
        stator_data[input_column], stator_data[C.TIME_MERGED], xvals=xvals, frac=1 / 500, return_sorted=False)

    stator_data = DataFrame([xvals, lowess_estimate]).T
    stator_data.columns = [C.TIME_MERGED.value, output_column.value]

    trace_data = pd.concat([trace_data, stator_data], axis=0)
    trace_data.sort_values(by=C.TIME_MERGED, inplace=True)
    # TODO: concat / interpolate probably leaves some uneven timestamps in here.

    return trace_data.interpolate()


def add_stator_d(trace_data, y_col=C.R_STATOR_TEMP, out_col=C.R_STATOR_TEMP_SMOOTH_D):
    """ Calculate derivative of stator temp to see if Tesla is limiting the rate of temp increase. """
    # verification:
    # 686.2, 99.49
    # 692,   101.91
    # (s),   (C)
    # Would be  .41 deg / s. That's quite small?
    #
    # 269.8  93.96
    # 275.8  102.80
    #  .5deg / s. It's not... 5/s is it?
    #
    # from vid:
    # 2:08  98
    # 2:15 102. So yeah makes sense, I'm just spending a long time with my foot in it.

    # stator_data = DataFrame([xvals, lowess_estimate]).T
    # stator_data.columns = [C.TIME_MERGED.value, C.R_STATOR_TEMP_SMOOTH.value]
    stator_data = trace_data[[C.TIME_MERGED, y_col]]

    stator_data[out_col] = numpy.gradient(
        stator_data[y_col],
        (stator_data[C.TIME_MERGED]))

    # Put the values in. All new rows since we estimated additional times.
    trace_data = pd.concat([trace_data, stator_data], axis=0)
    trace_data.sort_values(by=C.TIME_MERGED, inplace=True)
    trace_data.interpolate(inplace=True)

    return trace_data


def prune_repeat_samples(stator_data):
    class RunningEnumerator:
        last_set_time = -math.inf

        def check_and_set(self, time):
            if time >= self.last_set_time + .5:
                self.last_set_time = time
                return True
            return False

    e = RunningEnumerator()
    stator_data[C.R_STATOR_TEMP_SMOOTH.value] = [
        stator_temp if (not isnan(stator_temp) and e.check_and_set(time)) else NaN
        for stator_temp, time in zip(stator_data[C.R_STATOR_TEMP], stator_data[C.TIME_MERGED])]
    stator_data.dropna(inplace=True)


if __name__ == '__main__':
    main()

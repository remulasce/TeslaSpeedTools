"""
predict_torque.py: Adds torque predictions (incl. torque cut) to input files or traces

Run from the root directory (one level above this)
"""

import glob
from pathlib import Path

import numpy
import numpy as np
import pandas
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from data import C
import data
import torque_prediction_model


def main():
    print("Adding torque predictions to .csv files in predict_in")

    files = set(glob.glob("predict_in/*.csv"))
    files = files - set(glob.glob("predict_in/*.predict.csv"))
    print(f"Predicting for files: {files}")

    add_torque_predictions(files)
    print("Done")


def add_torque_predictions(files=[]):
    print("Processing files to add predictions...")

    for file in files:
        try:
            add_torque_prediction(Path(file))
        except Exception as e:
            print("Failed to process " + str(file) + " due to " + str(e))


def add_torque_prediction(file):
    print("Adding prediction to " + str(file))
    trace_data = data.read_files(file, clean=False)

    add_torque_prediction_trace(trace_data)
    trace_data.to_csv(file.with_suffix(".predict.csv"))


def add_torque_prediction_trace(trace_data):
    trace_data[C.PREDICTED_MAX_TORQUE] = torque_prediction_model.predict_torque_main_dict(
        trace_data,
        **torque_prediction_model.SRPLUS_TORQUE_MODEL_PARAMS)

    def predict_inner(frame):
        pred = frame[C.PREDICTED_MAX_TORQUE]
        torq = frame[C.R_TORQUE]
        accelerator_pedal = frame[C.ACCELERATOR_PEDAL]
        return max(0, 100 * (pred - torq) / pred) if accelerator_pedal > 50 else 0

    trace_data[C.PERCENT_TORQUE_CUT] = \
        trace_data.apply((lambda frame: predict_inner(frame)), axis=1)


def add_stator_estimates(file):
    print("Adding stator temp stuff")
    # Hmm, so this trace data has lots of n/a rows for stator temp if it wasn't read then.
    # I guess we could drop them and remerge on the time_merged?
    trace_data = pandas.read_csv(
        file, memory_map=True, na_values=''
    )
    trace_data = calculate_stator_estimates(trace_data)
    trace_data.to_csv(file + ".temp")


def calculate_stator_estimates(trace_data):
    local_data = trace_data[~trace_data[C.R_STATOR_TEMP].isnull()]
    temp_v_time = local_data[[C.R_STATOR_TEMP, C.TIME]]
    lowess_estimate = lowess(*temp_v_time.T.values.tolist(), frac=1 / 500, return_sorted=False)
    new_df = pd.DataFrame(np.transpose([temp_v_time[C.TIME], lowess_estimate]),
                          columns=[C.TIME.value, C.R_STATOR_TEMP_SMOOTH.value])
    # So the lowess is a sampled subset of available stator temp plots. To put it back in, we merge it
    # sql-style onto the rows we happened to sample, then interpolate it back over the input.
    trace_data = pd.merge(trace_data, new_df, how='outer', on=C.TIME)
    trace_data = trace_data.sort_values(C.TIME).interpolate()
    trace_data[C.R_STATOR_TEMP_D] = 10000 * numpy.gradient(trace_data[C.R_STATOR_TEMP_SMOOTH], (trace_data[C.TIME]))
    return trace_data


if __name__ == '__main__':
    main()

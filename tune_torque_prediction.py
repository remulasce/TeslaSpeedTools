import inspect

from plotly import subplots
from plotly_resampler import register_plotly_resampler
from scipy.optimize import curve_fit

import torque_prediction_model
from displays import show_figure, make_subplot_graph
from torque_prediction_model import *
from torque_prediction_model import SRPLUS_TORQUE_MODEL_PARAMS

register_plotly_resampler(mode='auto')


def main():
    print("Tuning torque prediction model....")

    model_files = TorquePredictionFiles.all
    print(f"Model files: {model_files}")

    print("Tuning params (check code for which parameters)")
    # Either tune a model, or review one
    model = train_all_params(files=model_files)
    # model = torque_prediction_model.SRPLUS_TORQUE_MODEL_PARAMS

    print("Got updated model. Checking...")
    validation_files = []
    validation_files.extend(TorquePredictionFiles.all)
    # validation_files.extend([TorquePredictionFiles.no_oil_cooler_fastlap,
    #                          TorquePredictionFiles.oil_cooler_with_normal_pump,
    #                          TorquePredictionFiles.oil_cooler_with_pump_overclocked
    #                          ])
    fig = review_prediction_trace(model, files=validation_files)
    print("Displaying")
    show_figure(fig)
    print("Done")


def train_all_params(files):
    trace_data = read_files(files)
    print("Training: Filtering out pedal under 75%")
    trace_data = filter_pedal_application(trace_data, pedal_min=75, pedal_max=100)
    print("Training: Filtering out speeds under field weakening")
    trace_data = filter_over_cliff(trace_data)

    return curve_fit_torque(trace_data, fn=tune_tqmax, guess=None)


def review_prediction_trace(model, files):
    titles = [str(file) for file in files for _ in range(1)]
    fig = subplots.make_subplots(rows=len(files), cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file)

        print("Validation: Filtering out pedal under 95")
        trace_data = filter_pedal_application(trace_data, pedal_min=95)

        # make_subplot_graph(fig, trace_data, x_axis=C.ACCELERATOR_PEDAL, y_axis="R torque", row=i, col=1, name=file)
        make_subplot_graph(fig, trace_data, x_axis=C.SPEED, y_axis=C.ACCELERATOR_PEDAL, row=i, col=1, color='gray')
        make_subplot_graph(fig, trace_data, x_axis="Speed", y_axis="R torque", row=i, col=1,
                           torque_estimate=lambda data: torque_prediction_model.predict_torque_frame(data, **model))

    fig.update_layout(height=len(files * 600))
    return fig


# Tuning methods:
# Subsets of tune_extended_params to allow fine-tuning of specific traces only.

def tune_all_params(
        data,
        tq_max,
        tq_accelerator_a,
        cliff_speed, cliff_v,
        fw_v_a, fw_accelerator_a,
        fw_a_hi, fw_b_hi, fw_c_hi, fw_d_hi,
        fw_a_lo, fw_b_lo, fw_c_lo, fw_d_lo,
):
    def fw_constants_from_params(*params):
        keys = ["fw_a", "fw_b", "fw_c", "fw_d"]
        return {key: value for (key, value) in zip(keys, params)}

    """
    Expands all sub-params for tuning.

        This function explicitly lists all known params for tuning.
    fit_curve can read the params here so it knows how many to tune.
    Only accepts pc-column DataFrames since we apply the column names here.

    Naturally, this is extremely slow. You should prefer tuning individual constants with sub-methods.
    """
    hi = fw_constants_from_params(fw_a_hi, fw_b_hi, fw_c_hi, fw_d_hi)
    lo = fw_constants_from_params(fw_a_lo, fw_b_lo, fw_c_lo, fw_d_lo)

    return predict_torque_main_dict(data,
                                    tq_max,
                                    tq_accelerator_a,
                                    cliff_speed, cliff_v,
                                    fw_v_a, fw_accelerator_a,
                                    hi, lo)


def tune_tqmax(data, tq_max):
    """
    Tunes just the tq_max for quicker development of the tuning stack.
    It should always be 470nm basically.
    """
    tuning = {
        "tq_max": tq_max
    }

    params = TorquePredictionModel.model_to_flat_dict(SRPLUS_TORQUE_MODEL_PARAMS)
    params.update(tuning)

    return tune_all_params(
        data,
        **params
    )


def tune_fw_all_log_constants_inner(trace_data,
                                    fw_v_a,
                                    fw_a_hi, fw_b_hi, fw_c_hi, fw_d_hi,
                                    fw_a_lo, fw_b_lo, fw_c_lo, fw_d_lo
                                    ):
    """
    Allows tuning of only fw constants.
    """
    params = SRPLUS_TORQUE_MODEL_PARAMS.copy()

    tuning = {
        "fw_v_a": fw_v_a,
        "fw_a_hi": fw_a_hi,
        "fw_b_hi": fw_b_hi,
        "fw_c_hi": fw_c_hi,
        "fw_d_hi": fw_d_hi,
        "fw_a_lo": fw_a_lo,
        "fw_b_lo": fw_b_lo,
        "fw_c_lo": fw_c_lo,
        "fw_d_lo": fw_d_lo,
    }

    params.update(tuning)

    return predict_torque_main_dict(
        trace_data,
        **params
    )


def curve_fit_torque(trace_data, fn=tune_all_params, guess=None):
    """
    P
    :param trace_data:
    :param fn:
    :param guess:
    :return:
    """
    if guess is None:
        guess = torque_prediction_model.SRPLUS_TORQUE_MODEL_PARAMS
    assert isinstance(guess, dict)

    flat_params = TorquePredictionModel.model_to_flat_dict(guess)
    fn_arg_names = inspect.getfullargspec(fn).args[1:]  # exempt the data param
    # Figure out which args are actually being tuned by this function. Tuning the full_expanded_params model
    # takes a long time, so the idea is you define a method which takes in only the parameters you want to tune.
    # That arg will call the full model using default parameters for the non-provided ones. Here, if a guess
    # isn't provided, we need to cut the default guess down to just the one that makes sense for the partial
    # model. So pull the params.
    guess = [flat_params[key] for key in fn_arg_names]

    trace_data = trace_data.sample(100)  # Maybe reconsider this
    all_frames = trace_data[[e.value for e in PC]].to_numpy().transpose()
    actual_torques = trace_data[C.R_TORQUE]

    print("Predicting curve...")
    # loss = 'cauchy', maxfev=
    popt = curve_fit(fn, all_frames, actual_torques, p0=guess, maxfev=100000, method='trf', loss='cauchy')

    print("Analysis results raw: " + str(popt[0]))

    params_list = list(popt[0])
    filled_flat_dict = TorquePredictionModel.model_to_flat_dict(
        torque_prediction_model.SRPLUS_TORQUE_MODEL_PARAMS)
    filled_flat_dict.update(zip(fn_arg_names, params_list))
    model = TorquePredictionModel.from_params_list(list(filled_flat_dict.values()))

    print("dict form: " + str(model))

    return model


if __name__ == '__main__':
    main()

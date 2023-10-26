import inspect

import numpy as np
from pandas import DataFrame
from scipy.optimize import curve_fit

import data
from data import *

# This should be the current best-fit, using hi/lo curves.
SRPLUS_TORQUE_MODEL_PARAMS = {
    'tq_max': 469.125, 'tq_accelerator_a': 6.612412744169433,
    'cliff_speed': 65.99973045750038, 'cliff_v': 0.0003816897486626024,
    'fw_v_a': 0.011468594400665411, 'fw_accelerator_a': 1.6758878532006707,
    # Would be fw_a_hi, fw_b_hi
    'hi': {'fw_a': -896.3746078952913, 'fw_b': -7.586995411250427,
           'fw_c': -1862.9018403847551, 'fw_d': 32.42677224719758},
    # Would be fw_a_lo, fw_b_lo
    'lo': {'fw_a': -1.1925343705085474, 'fw_b': 1.1119491977554612,
           'fw_c': 4.984857284976331, 'fw_d': 58.69134821315341}}

TESTING_PARAMS = {'tq_max': 469.125, 'tq_accelerator_a': 6.612402898684848, 'cliff_speed': 65.99948198831628,
                  'cliff_v': 0.0004365310967417108, 'fw_v_a': 0.0031821562832662098,
                  'fw_accelerator_a': 134.28248497551354,
                  'hi': -1091.7364583280012, 'lo': -7.586995411250677}

# So the fw_v_a didn't change, probably because it's not actually included/necessary. Mostly fw_b and fw_d.
# We can get rid of it entirely if we have a whole different curve. OR rather, it should represent the proportion
# between the two now.
HV_FW_CURVE = [-897.36050249, -7.58699632, -1862.4732444, 32.42677225]
LV_FW_CURVE = [-1.10960402, 0.45828758, 5.35433467, 55.45196733]


def model_from_params_list(params, pattern=SRPLUS_TORQUE_MODEL_PARAMS):
    # print(f"{params}, {pattern}}")
    indict = {}
    for (key, value) in pattern.items():
        if isinstance(value, dict):
            indict[key] = (model_from_params_list(params, value))
        else:
            # print(f"orig key: {key}")
            # print(f"{key}: {params[0]}")
            indict[key] = params.pop(0)
    return indict


def flat_model_from_params(params):
    keys = flat_dict(SRPLUS_TORQUE_MODEL_PARAMS).keys()
    return {key: value for (key, value) in zip(keys, params)}


def fw_constants_from_params(*params):
    keys = ["fw_a", "fw_b", "fw_c", "fw_d"]
    return {key: value for (key, value) in zip(keys, params)}


def flat_dict(params_dict, suffix=None):
    indict = {}
    for (key, value) in params_dict.items():
        if isinstance(value, dict):
            indict.update(flat_dict(value, key))
        else:
            indict[key + "_" + suffix if suffix is not None else key] = value
    return indict


class PC(str, Enum):
    R_TORQUE = C.R_TORQUE.value,
    ACCELERATOR_PEDAL = C.ACCELERATOR_PEDAL.value,
    SPEED = C.SPEED.value,
    BATTERY_VOLTAGE = C.BATTERY_VOLTAGE.value,


# Predicts torque directly from _PARAMS. Doesn't work directly in the tuner because
# it uses nested values in dict.
def predict_torque_main_dict(
        data,
        tq_max,
        tq_accelerator_a,
        cliff_speed, cliff_v,
        fw_v_a, fw_accelerator_a,
        hi, lo,

):
    """
    Predict torque, given standard bundled constants format.
    """

    if not isinstance(data, DataFrame):
        data = DataFrame.from_records(data.T, columns=[e.value for e in PC])
    return data.apply(lambda frame: predict_torque_frame(
        frame,
        tq_accelerator_a=tq_accelerator_a, tq_max=tq_max,
        cliff_speed=cliff_speed, cliff_v=cliff_v,
        fw_v_a=fw_v_a, fw_accelerator_a=fw_accelerator_a,
        hi=hi, lo=lo
    ), 1)


def predict_torque_frame(frame, **kwargs):
    """ Lines up the various inner-methods for each row's prediction"""
    return split_torque_cliff(
        frame,
        predict_limited_torque(**kwargs),
        predict_field_weakening_torque(
            predict_field_weakening_max_torque_split(**kwargs),
            **kwargs
        ), **kwargs)


def split_torque_cliff(frame,
                       torque_limited_curve, field_weakening_curve, cliff_speed, cliff_v, **_):
    """
     Structure for torque prediction. Below speed torque_cliff_speed, torque follows a torque-limited curve.
     Above that speed, torque is predicted by a field weakening function.

     Likely this cliff speed function isn't needed; it should be predictable as min(maxtorque, field_weakening)
    """
    speed = frame[PC.SPEED] + cliff_v * frame[PC.BATTERY_VOLTAGE]
    if speed < cliff_speed:
        return torque_limited_curve(frame)
    else:
        return field_weakening_curve(frame)


def predict_limited_torque(tq_max, tq_accelerator_a, **_):
    return lambda f: min(accelerator_coefficient(f[C.ACCELERATOR_PEDAL], tq_accelerator_a), tq_max)


def predict_field_weakening_torque(max_fw_torque_fn, fw_accelerator_a, **_):
    return lambda f: \
        min(accelerator_coefficient(f[C.ACCELERATOR_PEDAL], fw_accelerator_a), 120) * \
        max_fw_torque_fn(f)


def accelerator_coefficient(accelerator_pedal, accelerator_a, zero_point=15):
    return (accelerator_pedal - zero_point) * accelerator_a


def predict_field_weakening_max_torque_split(**kwargs):
    return lambda f: predict_fw_max_split_inner(f, **kwargs)


def predict_fw_max_split_inner(f, fw_v_a, hi, lo, **_):
    fw_v_a = voltage_coefficient(f[C.BATTERY_VOLTAGE], fw_v_a)
    # fw_v_a = 0
    lv = fw_v_a
    hv = 1 - fw_v_a

    return lv * predict_field_weakening_max_tq(**lo)(f) + \
           hv * predict_field_weakening_max_tq(**hi)(f)


# y = a log (b(x-d)) + c
def predict_field_weakening_max_tq(fw_a, fw_b, fw_c, fw_d, **_):
    return lambda f: (fw_c + fw_a * np.log(
        max(
            0.1,
            fw_b *
            (f[C.SPEED] - fw_d))))


def voltage_coefficient(voltage, fw_v_a):
    nominal_v = 400
    return (nominal_v - fw_v_a * voltage) / nominal_v


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

    params = flat_dict(SRPLUS_TORQUE_MODEL_PARAMS)
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
        guess = SRPLUS_TORQUE_MODEL_PARAMS
    assert isinstance(guess, dict)

    flat_params = flat_dict(guess)
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
    filled_flat_dict = flat_dict(SRPLUS_TORQUE_MODEL_PARAMS)
    filled_flat_dict.update(zip(fn_arg_names, params_list))
    model = model_from_params_list(list(filled_flat_dict.values()))

    print("dict form: " + str(model))

    return model


if __name__ == '__main__':
    print("Models.py doesn't do anything on its own. Use tune_torque_preduction.py")
    print("Done.")

import numpy as np
from pandas import DataFrame

from data import *


class TorquePredictionModel(dict):
    pass


# As presented in https://www.youtube.com/watch?v=Vzss8orfwi0
SRPLUS_TORQUE_MODEL_PARAMS_VIDEO_VERSION = TorquePredictionModel({
    'tq_max': 469.125,
    'tq_accelerator_a': 6.612412744169433,
    # TODO: Torque-limit should just be a constant applied on top of the curve output. No need to _also_ model the speed
    'cliff_speed': 65.99973045750038, 'cliff_v': 0.0003816897486626024,
    # Field weakening curve definition. Field weakening torque is heavily dependent on battery voltage. It's modelled
    # here as a linear interpolation between a 'hi' and 'lo' voltage curve. fw_v_a defines the blend somehow
    'fw_v_a': 0.011468594400665411,
    'hi': {'fw_a': -896.3746078952913, 'fw_b': -7.586995411250427,
           'fw_c': -1862.9018403847551, 'fw_d': 32.42677224719758},
    'lo': {'fw_a': -1.1925343705085474, 'fw_b': 1.1119491977554612,
           'fw_c': 4.984857284976331, 'fw_d': 58.69134821315341},
    # Tries to derate for accelerator position. Doesn't really work.
    'fw_accelerator_a': 1.6758878532006707,
})


class PC(str, Enum):
    """
    Just the columns needed out of data to predict the torque.
    Redefined in here because sometimes the dataframes don't line up exactly.
    """
    R_TORQUE = C.R_TORQUE.value,
    ACCELERATOR_PEDAL = C.ACCELERATOR_PEDAL.value,
    SPEED = C.SPEED.value,
    BATTERY_VOLTAGE = C.BATTERY_VOLTAGE.value,


def model_from_params_list(params: list, pattern: dict = SRPLUS_TORQUE_MODEL_PARAMS_VIDEO_VERSION):
    indict = {}
    for (key, value) in pattern.items():
        if isinstance(value, dict):
            indict[key] = (model_from_params_list(params, value))
        else:
            indict[key] = params.pop(0)
    return indict


def flat_model_from_params(params):
    keys = flat_dict(SRPLUS_TORQUE_MODEL_PARAMS_VIDEO_VERSION).keys()
    return {key: value for (key, value) in zip(keys, params)}


def fw_constants_from_params(*params):
    keys = ["fw_a", "fw_b", "fw_c", "fw_d"]
    return {key: value for (key, value) in zip(keys, params)}


def flat_dict(params_dict, suffix=None):
    """
    Converts a nested dict (human-readable-ish) of params into a flat one.
    :param params_dict:
    :param suffix: For recursion only, just provide None.
    """
    indict = {}
    for (key, value) in params_dict.items():
        if isinstance(value, dict):
            indict.update(flat_dict(value, key))
        else:
            indict[key + "_" + suffix if suffix is not None else key] = value
    return indict


# Predicts torque directly from _PARAMS. Doesn't work directly in the tuner because
# it uses nested values in dict.
def predict_torque_main_dict(
        data,
        tq_max,
        tq_accelerator_a,
        cliff_speed, cliff_v,
        fw_v_a, fw_accelerator_a,
        hi, lo
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


if __name__ == '__main__':
    print("Models.py doesn't do anything on its own. Use tune_torque_preduction.py")
    print("Done.")

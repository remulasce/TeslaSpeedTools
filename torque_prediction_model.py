from __future__ import annotations
import numpy as np
from pandas import DataFrame

from data import *


class TorquePredictionModel(dict):
    """
    Add a little bit of sanity to this dict stuff
    """
    example_dict = {
        'tq_max': 469.125,
        'tq_accelerator_a': 6.612412744169433,
        # Field weakening curve definition. Field weakening torque is heavily dependent on battery voltage. It's
        # modelled here as a linear interpolation between a 'hi' and 'lo' voltage curve. fw_v_a defines the blend
        # somehow
        'fw_v_a': 0.011468594400665411,
        'hi': {'fw_a': -896.3746078952913, 'fw_b': -7.586995411250427,
               'fw_c': -1862.9018403847551, 'fw_d': 32.42677224719758},
        'lo': {'fw_a': -1.1925343705085474, 'fw_b': 1.1119491977554612,
               'fw_c': 4.984857284976331, 'fw_d': 58.69134821315341},
        # Tries to derate for accelerator position. Doesn't really work.
        'fw_accelerator_a': 1.6758878532006707,
    }

    @staticmethod
    def from_params_list(params: list, pattern: dict = example_dict) -> TorquePredictionModel:
        """
        Constructs a TorquePredictionModel from a flat list of individual params, which must have the same count
        as the number of recursed params in the pattern.
        """
        indict = {}
        for (key, value) in pattern.items():
            if isinstance(value, dict):
                indict[key] = TorquePredictionModel.from_params_list(params, value)
            else:
                indict[key] = params.pop(0)
        return TorquePredictionModel(indict)

    @staticmethod
    def flat_dict_from_params(params: list) -> dict:
        keys = params.model_to_flat_dict(SRPLUS_TORQUE_MODEL_PARAMS).keys()
        return {key: value for (key, value) in zip(keys, params)}

    @staticmethod
    def model_to_flat_dict(model: TorquePredictionModel, suffix=None):
        """
        Converts a nested dict (human-readable-ish) of params into a flat one.
        :param model:
        :param suffix: For recursion only, just provide None.
        """
        indict = {}
        for (key, value) in model.items():
            if isinstance(value, dict):
                indict.update(model.model_to_flat_dict(value, key))
            else:
                indict[key + "_" + suffix if suffix is not None else key] = value
        return indict


# Simplified/improved without explicit torque cliff. Still not perfect.
# Based on     training_set = [th_1, th_3, th_5, th_6, no_oil_cooler_fastlap]. It seems some of these aren't compatible
# really.
SRPLUS_TORQUE_MODEL_PARAMS = TorquePredictionModel(
    {'tq_max': 470.23825994385976, 'tq_accelerator_a': 6.612412744169433, 'fw_v_a': 0.01269308238279221,
     'hi': {'fw_a': -903.2139178114219, 'fw_b': -7.586995752266236, 'fw_c': -1859.9319144396932,
            'fw_d': 32.42677224682448},
     'lo': {'fw_a': -1.0879315266950822, 'fw_b': 1.5855009105173516, 'fw_c': 4.445505561635066,
            'fw_d': 61.62181080585337}, 'fw_accelerator_a': 1.617242764918341
     })

# As presented in https://www.youtube.com/watch?v=Vzss8orfwi0
SRPLUS_TORQUE_MODEL_PARAMS_VIDEO_VERSION = TorquePredictionModel({
    'tq_max': 469.125,
    'tq_accelerator_a': 6.612412744169433,
    # Not used any more, it's just a hard cap of torque now.
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


# Predicts torque directly from _PARAMS. Doesn't work directly in the tuner because
# it uses nested values in dict.
def predict_torque_main_dict(
        data,
        tq_max,
        tq_accelerator_a,
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
                       torque_limited_curve, field_weakening_curve, **_):
    """
    Torque is limited either by the field-weakening characteristics with falloff due to speed and voltage, or the
    fixed inverter amperage limit when at lower speeds.
    """
    return min(torque_limited_curve(frame), field_weakening_curve(frame))


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

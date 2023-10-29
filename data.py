import glob
import os
from enum import Enum
from pathlib import Path

import pandas


class C(str, Enum):
    # ScanMyTesla erroneously reports some values in the SR+ as "front", even though it's single-motor.
    R_POWER = "R power"
    R_TORQUE = "R torque"
    F_TORQUE = "F torque",
    SPEED = "Speed",
    BATTERY_VOLTAGE = "Battery voltage",
    BATTERY_CURRENT = "Battery current",
    ACCELERATOR_PEDAL = "Accelerator Pedal",
    BVOLTAGE_MINUS_80 = "BVoltage Minus 80",
    BATTERY_INLET = 'Battery inlet',
    POWERTRAIN_INLET = 'Powertrain inlet',
    F_STATOR_TEMP = 'F Stator temp',
    R_STATOR_TEMP = 'R Stator temp',
    R_STATOR_TEMP_SMOOTH = 'R Stator temp smooth',
    R_STATOR_TEMP_D = 'R Stator temp D'
    R_STATOR_TEMP_SMOOTH_D = 'R Stator temp smooth D'
    FRONT_OIL_FLOW = 'Front oil flow',
    CELL_TEMP_MAX = 'Cell temp max',
    CELL_TEMP_MID = 'Cell temp mid',
    FRONT_OIL_TEMP = 'Front oil temp',
    R_HEAT_SINK = 'R heat sink',
    AC_COMPRESSOR_DUTY = 'A/C compressor duty',
    BATTERY_FLOW = 'Battery flow',
    POWERTRAIN_FLOW = 'Powertrain flow',
    R_INVERTER_PCB_TEMP = 'R Inverter PCB temp',
    R_INVERTER_TEMP = 'R Inverter temp',
    MAX_DISCHARGE_POWER = "Max discharge power",
    PREDICTED_MAX_TORQUE = "Predicted Max Torque",
    PERCENT_TORQUE_CUT = "Percent Torque Cut"
    POWER_PENALTY = "Power Penalty"
    TIME = "Time"
    TIME_MERGED = "Time Merged"
    TARGET_PT_ACTIVECOOL = "Target PT ActiveCool",
    TARGET_PT_PASSIVECOOL = "Target PT Passive",
    TARGET_BAT_ACTIVECOOL = "Target bat ActiveCool",
    TARGET_BAT_PASSIVECOOL = "Target bat Passive"


class Files:
    # Inverter failure lmao
    inverter_failure = Path("traces/inverter_failure/inverter_2.csv")

    # First time at East in the Tesla, 2022-11-21
    # Unthrottled used to train model for 100mph+
    east_unthrottled = Path("traces/2022-11-22_East/east_unthrottled.csv")
    east_2_throttled = Path("traces/2022-11-22_East/east_2_throttled.csv")

    # Rainy day; final session was 1:25.9, no cooling box.
    west_fast = "West_Perf 2022-12-10 15-42-44.csv"
    west_fast_unthrottled = Path("traces/2022-12-10_West/west_unthrottled.csv")

    # Experimental tests of pulls from 40-80mph at different SOCs
    # Also some varied throttle tests
    th_1 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 00-14-47.csv")  # high volt test
    th_2 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 00-22-56.csv")
    th_3 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 00-32-50.csv")
    th_4 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 00-42-14.csv")
    th_5 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 00-48-39.csv.bad")
    th_6 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 01-04-54.csv")  # low volt test
    th_7 = Path("traces/torque_soc_experiment_night/Perf 2022-12-13 01-10-45.csv")
    th_all = [th_1, th_2, th_3, th_4, th_6, th_7]

    # Ongrid Buttonwillow 2023-1-8 with T9
    # _4: Fastlap
    bw_23_1_8_S4 = Path("traces/2023-1-9_Buttonwillow/Perf 2023-01-08 14-19-03.csv")
    bw_23_1_8_S4_unthrottled = Path("traces/2023-1-9_Buttonwillow/bw_unthrottled_4.csv")
    # _5: With pass, 2:02 or something
    bw_23_1_8_S5 = Path("traces/2023-1-9_Buttonwillow/Perf 2023-01-08 15-35-29.csv")
    bw_23_1_8_S5_unthrottled = Path("traces/2023-1-9_Buttonwillow/bw_unthrottled_5.csv.csv")
    bw_23_1_8_all = glob.glob("traces/2023-1-9_Buttonwillow/*.csv")

    bw_200_fastlap_trace = Path("traces/2023-2-18_bwtesting/Perf 2023-02-18 11-01-55.csv")

    cal_precondition = Path("traces/cal_precondition/Perf 2023-03-04 00-53-01.csv")

    bw_oil_S1 = Path("traces/Oil Cooler Test Day/Perf 2023-05-27 08-01-46.csv")
    bw_oil_SC = Path("traces/Oil Cooler Test Day/Perf 2023-05-27 11-21-59.csv")
    bw_oil_S5 = Path("traces/Oil Cooler Test Day/Perf 2023-05-27 15-23-14.csv")


class TorquePredictionFiles:
    pred_folder = Path("torque_prediction_traces")

    def add_folder(self, filename):
        return self.pred_folder + Path(filename)

    all = []
    for name in os.listdir(pred_folder):
        all.append(pred_folder.joinpath(Path(name)))

    th_1 = Path("torque_prediction_traces/Perf 2022-12-13 00-14-47.csv")  # high volt test
    th_3 = Path("torque_prediction_traces/Perf 2022-12-13 00-32-50.csv")
    th_5 = Path("torque_prediction_traces/Perf 2022-12-13 00-48-39.csv")
    th_6 = Path("torque_prediction_traces/Perf 2022-12-13 01-04-54.csv")  # low volt test

    no_oil_cooler_fastlap = Path("example_traces/no_oil_cooler_fastlap.csv")
    oil_cooler_with_pump_overclocked = Path("example_traces/oil_cooler_with_pump_overclocked.csv")
    oil_cooler_with_normal_pump = Path("example_traces/oil_cooler_with_stock_pump.csv")

    training_set = [th_1, th_3, th_5, th_6, no_oil_cooler_fastlap]


colors_map = {
    C.PERCENT_TORQUE_CUT: "red",
    C.SPEED: "black",

    # Acceleration: Black and near-black
    C.PREDICTED_MAX_TORQUE: "grey",
    C.F_TORQUE: "black",
    C.R_TORQUE: "black",

    # "Hot" side: Reds
    C.R_STATOR_TEMP: "red",
    C.R_STATOR_TEMP_SMOOTH: "red",
    C.R_STATOR_TEMP_D: "rgb(200, 0, 0)",
    C.R_STATOR_TEMP_SMOOTH_D: "rgb(200, 0, 0)",
    C.FRONT_OIL_TEMP: "orangered",
    C.FRONT_OIL_FLOW: "rgb(200, 120, 60)",

    C.CELL_TEMP_MID: "rgb(0, 30, 160)",

    # "Cold" side temps: Blues
    C.POWERTRAIN_INLET: "blue",
    C.BATTERY_INLET: "blue",
    C.BATTERY_FLOW: "rgb(0, 30, 200)",
    C.POWERTRAIN_FLOW: "rgb(0, 30, 200)",

    C.R_TORQUE: "black",
    C.R_POWER: "black",
}


def filter_80_mph(trace_data):
    # Only around 80mph, analyse throttle response. 80mph = 128km/h
    return trace_data[
        (trace_data[C.SPEED] > 123) &
        (trace_data[C.SPEED] < 133) &
        (trace_data[C.ACCELERATOR_PEDAL] < 90) &  # Trendline doesn't fit at the top.
        (trace_data[C.ACCELERATOR_PEDAL] > 5)
        ]


def filter_under_cliff(trace_data):
    # Roughly matches the torque limited portion.
    return trace_data[(trace_data[C.SPEED] < 66)]


def filter_over_cliff(trace_data):
    """
    Removes most speeds where torque is limited to max, since it's always gunna be 469
    """
    return trace_data[(trace_data[C.SPEED] > 66)]


def filter_over_speed(trace_data, speed):
    """
    More generic named filter. Useful to get out the drive from the paddock to the hotpit.
    """
    return trace_data[(trace_data[C.SPEED] > speed)]


def filter_pedal_application(trace_data, pedal_min=75, pedal_max=100):
    return trace_data[
        (trace_data["Accelerator Pedal"] >= pedal_min) &
        (trace_data["Accelerator Pedal"] <= pedal_max)
        ]


def read_trace_json(raw, clean=True, interpolate=True):
    df = pandas.read_json(raw)
    return process_trace(df, raw, clean, interpolate)


def process_trace(df, interpolate=True):
    if interpolate:
        df.interpolate(inplace=True, limit_direction='both', method='linear')

    merge_time(df)
    if interpolate:
        df.interpolate(inplace=True)
        assert not df.isnull().values.any()
    return df


def read_files(files, clean=True, interpolate=True):
    if type(files) is not list:
        files = [files]

    dfs = []
    for file in [Path(f) for f in files]:
        df = pandas.read_csv(file, memory_map=True, na_values='')
        if interpolate:
            df.interpolate(inplace=True, limit_direction='both', method='linear')
        if clean:
            if Files.th_1 == file:
                df = df[df[C.SPEED] > 62]  # Filter some nonusable segments
            if Files.th_3 == file:
                # TC cut
                df = df.loc[(df[C.SPEED] < 48) | (df[C.SPEED] > 64)]
        dfs.append(df)

    trace_data = pandas.concat(dfs, axis=0, ignore_index=True)
    merge_time(trace_data)

    if interpolate:
        trace_data.interpolate(inplace=True, limit_direction='both', method='linear')
        assert not trace_data.isnull().values.any()

    return trace_data


def merge_time(trace_data):
    trace_data["Time Merged"] = [min(1, abs(dt / 1000)) for dt in trace_data["Time"].diff()]
    trace_data["Time Merged"] = trace_data["Time Merged"].cumsum()
    trace_data.dropna(axis=1, how='all', inplace=True)

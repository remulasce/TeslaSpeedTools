""" filtering.py: Selects subsections of trace files or frames for modelling.

 Also includes generic filter methods.

 """
import data


def in_any(values, ranges=[]):
    """ ranges is list of tuple (min, max) """
    return [
        any(lower <= value <= upper for (lower, upper) in ranges)
        for value in values]


def select_bw_unthrottled_laps():
    print("bw unthrottled filter")
    trace_data_4 = data.read_files(data.Files.bw_23_1_8_S4)
    trace_data_4 = trace_data_4[
        trace_data_4["Time"] < 300e3
        ]

    trace_data_4.to_csv("bw_unthrottled_4.csv")

    trace_data_5 = data.read_files(data.Files.bw_23_1_8_S5)
    trace_data_5 = trace_data_5[
        (trace_data_5["Time"] > 300e3) &
        (trace_data_5["Time"] < 540e3)
        ]

    trace_data_5.to_csv("bw_unthrottled_5.csv")


def select_west_unthrottled_laps():
    print("Filtering West unthrottled")
    trace_data = data.read_files(data.Files.west_fast, clean=False)
    trace_data = trace_data[
        (trace_data["Time"] > 40e3) & (trace_data["Time"] < 300e3)]
    trace_data = trace_data[
        in_any(trace_data["Speed"], [(0, 42), (65, 200)])
    ]

    trace_data.to_csv("west_unthrottled.csv")


def select_east_unthrottled_laps():
    print("Filtering East unthrottled")
    trace_data = data.read_files(data.Files.east_1, clean=False)
    trace_data = trace_data[
        (trace_data["Time"] > 86e3) & (trace_data["Time"] < 360e3)]

    trace_data.to_csv("east_unthrottled.csv")
from plotly import subplots
from plotly_resampler import register_plotly_resampler

import models
from data import TorquePredictionFiles, read_files, filter_pedal_application, filter_over_cliff, C
from displays import show_figure, make_subplot_graph

register_plotly_resampler(mode='auto')


def main():
    print("Tuning torque prediction model....")

    model_files = TorquePredictionFiles.all
    print(f"Model files: {model_files}")

    print("Tuning params (check code for which parameters)")
    # Either tune a model, or review one
    # model = tune_all_params(files=model_files)
    model = models.SRPLUS_TORQUE_MODEL_PARAMS

    print("Got updated model. Checking...")
    validation_files = [TorquePredictionFiles.no_oil_cooler_fastlap,
                        TorquePredictionFiles.oil_cooler_with_normal_pump,
                        TorquePredictionFiles.oil_cooler_with_pump_overclocked
                        ] + model_files
    # validation_files = [TorquePredictionFiles.no_oil_cooler_fastlap,
    #                     TorquePredictionFiles.oil_cooler_with_normal_pump,
    #                     TorquePredictionFiles.oil_cooler_with_pump_overclocked
    #                     ]
    fig = review_prediction_trace(model, files=validation_files)
    show_figure(fig)
    print("Done")


def tune_all_params(files):
    trace_data = read_files(files)
    print("Filtering out pedal under 75%")
    trace_data = filter_pedal_application(trace_data, pedal_min=75, pedal_max=100)
    print("Filtering out speeds under field weakening")
    trace_data = filter_over_cliff(trace_data)

    return models.curve_fit_torque(trace_data, guess=None)


def review_prediction_trace(model, files):
    titles = [str(file) for file in files for _ in range(1)]
    fig = subplots.make_subplots(rows=len(files), cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=titles)

    for file, i in zip(files, range(1, len(files) + 1)):
        trace_data = read_files(file)

        print("filtering out pedal under 95")
        trace_data = filter_pedal_application(trace_data, pedal_min=95)

        # make_subplot_graph(fig, trace_data, x_axis=C.ACCELERATOR_PEDAL, y_axis="R torque", row=i, col=1, name=file)
        make_subplot_graph(fig, trace_data, x_axis=C.SPEED, y_axis=C.ACCELERATOR_PEDAL, row=i, col=1, color='gray')
        make_subplot_graph(fig, trace_data, x_axis="Speed", y_axis="R torque", row=i, col=1,
                           torque_estimate=lambda data: models.predict_torque_frame(data, **model))

    fig.update_layout(height=len(files * 600))
    return fig


if __name__ == '__main__':
    main()

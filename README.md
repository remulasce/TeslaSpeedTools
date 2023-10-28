# TeslaSpeedTools

Analysis and graphing for performance-related analysis of Tesla cars.

Specifically my Model 3 SR+, using ScanMyTesla logs.


## Features 
This is partially just a dump of separate features. Major ones:

* Graphing of speed, voltages, etc. Use dash_test.py
* Predict how much torque the car _should_ be making under current conditions (speed, voltage), for purpose of evaluating thermal restrictions.
  * Automatically added to dash_test.py analysis
  * Add predictions to a standalone .csv file adding the trace to predict_in folder and using shortcuts therein.
* Produce the torque predictions
  * See `tune_torque_prediction.py` for this workflow
  * The model itself is in `torque_prediction_model.py`


## Tuning Torque Predictions
Use `main.py` to invoke tests against `models.py`. `models.py` has the code to predict and tune predictions, and also 
keeps the "main" prediction in `VOLTAGE_MODELLED_PARAMS_FW`.

The loop is to   

## Development
Suggest using Intellij PyCharm. There should be a Conda environment set up with the dependencies in there.


### Display Server
The project uses Plotly and Plotly Dash. Originally, plain Plotly was used, eg. in the analyze_session and tune_torque files.
Plotly Dash is used for a more full-featured experience, eg. drag and drop files.
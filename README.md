# TeslaSpeedTools

Analysis and graphing for performance-related analysis of Tesla cars.

Specifically my Model 3 SR+, using ScanMyTesla logs.


## Features 
This is partially just a dump of separate features. Major ones:

* Graphing of speed, voltages, etc. Use dash_test.py
* Predict how much torque the car _should_ be making under current conditions (speed, voltage), for purpose of evaluating thermal restrictions.
  * Automatically added to dash_test.py analysis
* Produce the torque predictions
  * See functions in models.py


## Development
Suggest using Intellij PyCharm. There should be a Conda environment set up with the dependencies in there.
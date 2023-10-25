echo "Adding predictions to telemetry files"
cd ..
conda run --no-capture-output -n PloTesler python predict_torque.py
wait
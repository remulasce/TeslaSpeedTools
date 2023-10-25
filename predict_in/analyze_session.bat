echo "Analyzing session..."
cd ..
start powershell -NoExit -Command "conda run --no-capture-output -n PloTesler python analyze_session.py \"%*\""
pause
start "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://127.0.0.1:8050/
pause
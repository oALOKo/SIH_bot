# SIH_bot: Liveness Detection Application

This repository contains a full-stack liveness detection application designed to operate as a Chrome extension with a Python backend.

## Structure

* **`ML_backend/`**: Contains the Flask API (`app.py`), the YOLO and temporal CNN models for liveness inference, and related scripts.
* **`front_end/`**: Contains the Chrome Extension source code (`manifest.json`, `background.js`, `content.js`) that captures the user's webcam and queries the backend.
* **`requirements.txt`**: The Python dependencies required to run the ML backend.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the backend:
   ```bash
   cd ML_backend
   python app.py
   ```
3. Load the extension in Chrome:
   - Navigate to `chrome://extensions/`
   - Enable **Developer mode**
   - Click **Load unpacked** and select the `front_end` folder.

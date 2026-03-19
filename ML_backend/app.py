import cv2
import torch
import pickle
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from collections import deque
from ultralytics import YOLO
from preprocess import frame_transform
from model import LivenessModel, TCN, TemporalBlock
import concurrent.futures
import asyncio
import websockets
import threading

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

yolo_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
with open(os.path.join(BASE_DIR, "liveness_final.pkl"), "rb") as file:
    liveness_model = pickle.load(file)
liveness_model.eval()

frame_buffer = deque(maxlen=8)
skip_frames = 2  # Skip YOLO processing every nth frame
frame_count = 0
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def detect_faces_async(frame):
    return yolo_model(frame)[0].boxes.xyxy.cpu().numpy()

def preprocess_for_liveness(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_transform(frame_rgb)

def process_webcam_frames():
    global frame_count
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 24)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    prediction, label_color = "Waiting...", (255, 255, 255)
    face_boxes = None  # Cache face detection results
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames == 0 or face_boxes is None:
            future = executor.submit(detect_faces_async, frame)
            face_boxes = future.result()
        
        if len(face_boxes) != 1:
            cv2.putText(frame, "Invalid frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_buffer.clear()
            prediction, label_color = "Attack", (0, 0, 255)  # Set prediction to "Attack"
        else:
            x1, y1, x2, y2 = map(int, face_boxes[0][:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            processed_frame = preprocess_for_liveness(frame)
            frame_buffer.append(processed_frame)
            
            if len(frame_buffer) == 8:
                input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0)
                with torch.no_grad():
                    output = liveness_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    attack_prob = probabilities[0, 1].item()
                prediction, label_color = ("Attack", (0, 0, 255)) if attack_prob > 0.65 else ("Real", (0, 255, 0))
                frame_buffer.clear()
        
        cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_encoded = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_encoded + b"\r\n")
    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_liveness", methods=["POST"])
def check_liveness():
    if 'frame' not in request.files:
        return jsonify({"status": "Error", "message": "No frame provided"}), 400
    
    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"status": "Error", "message": "Invalid image"}), 400

    face_boxes = detect_faces_async(frame)
    if len(face_boxes) != 1:
        return jsonify({"status": "Attack"})
    
    processed_frame = preprocess_for_liveness(frame)
    input_tensor = torch.stack([processed_frame] * 8).unsqueeze(0)
    
    with torch.no_grad():
        output = liveness_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        attack_prob = probabilities[0, 1].item()
    
    prediction = "Attack" if attack_prob > 0.65 else "Real"
    return jsonify({"status": prediction})

@app.route("/video_feed")
def video_feed():
    return Response(process_webcam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, threaded=True, use_reloader=False)

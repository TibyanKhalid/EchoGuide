# app.py

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import json
import threading
from queue import Queue
import time
from ultralytics import YOLO
from yolo_detection.spatial_utils import get_position_description, sort_by_importance
from yolo_detection.navigation_classes import NAVIGATION_CLASSES, PRIORITY_MAP
from gtts import gTTS
import tempfile
import os

app = Flask(__name__)
CORS(app)

class WebNavigationAssistant:
    def __init__(self):
        print("[Init] Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        self.conf_threshold = 0.4
        self.narration_interval = 5
        self.last_narration_time = 0
        
        self.current_frame = None
        self.current_detections = []
        self.current_narration = "Waiting for camera..."
        
        # Camera
        self.camera = cv2.VideoCapture(0)
        self.camera_lock = threading.Lock()
        
        print("[Init] Ready!")
    
    def process_frame(self, frame):
        """Process frame with YOLO"""
        results = self.yolo_model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        height, width = frame.shape[:2]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            if class_name in NAVIGATION_CLASSES:
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox.tolist(),
                    'size': bbox_area,
                    'position': get_position_description(bbox, width, height)
                })
        
        return detections
    
    def generate_narration(self, detections):
        """Generate simple narration (Milestone 1 style)"""
        if not detections:
            return "No obstacles detected"
        
        # Sort by importance
        sorted_dets = sort_by_importance(detections, PRIORITY_MAP)
        
        # Take top 3 most important objects
        top_detections = sorted_dets[:3]
        
        descriptions = []
        for det in top_detections:
            desc = f"{det['class']} {det['position']}"
            descriptions.append(desc)
        
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]}, and {descriptions[1]}"
        else:
            return f"{descriptions[0]}, {descriptions[1]}, and {descriptions[2]}"
    
    def generate_audio(self, text):
        """Generate audio file from text using gTTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to static folder
            audio_path = os.path.join('static', 'current_narration.mp3')
            tts.save(audio_path)
            
            return audio_path
        except Exception as e:
            print(f"[ERROR] Audio generation failed: {e}")
            return None
    
    def get_frame(self):
        """Get current camera frame"""
        with self.camera_lock:
            success, frame = self.camera.read()
            if not success:
                return None
            
            # Process detections
            self.current_detections = self.process_frame(frame)
            
            # Draw bounding boxes
            for det in self.current_detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} ({det['confidence']:.2f})"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if should generate new narration
            current_time = time.time()
            if current_time - self.last_narration_time >= self.narration_interval:
                self.last_narration_time = current_time
                self.current_narration = self.generate_narration(self.current_detections)
                print(f"[NARRATION]: {self.current_narration}")
                self.generate_audio(self.current_narration)
            
            self.current_frame = frame
            return frame
    
    def release(self):
        """Release camera"""
        self.camera.release()


# Global assistant instance
assistant = WebNavigationAssistant()


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


def generate_frames():
    """Video streaming generator"""
    while True:
        frame = assistant.get_frame()
        if frame is None:
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_narration')
def get_narration():
    """Get current narration text"""
    return jsonify({
        'narration': assistant.current_narration,
        'detection_count': len(assistant.current_detections),
        'detections': [
            {
                'class': d['class'],
                'position': d['position'],
                'confidence': round(d['confidence'], 2)
            }
            for d in assistant.current_detections
        ]
    })


@app.route('/get_audio')
def get_audio():
    """Get current audio file"""
    audio_path = 'static/current_narration.mp3'
    if os.path.exists(audio_path):
        return jsonify({'audio_url': '/' + audio_path})
    return jsonify({'audio_url': None})


@app.route('/set_interval', methods=['POST'])
def set_interval():
    """Update narration interval"""
    data = request.get_json()
    interval = data.get('interval', 5)
    assistant.narration_interval = max(3, min(15, interval))  # Clamp between 3-15 seconds
    print(f"[SETTING] Narration interval changed to {assistant.narration_interval}s")
    return jsonify({'success': True, 'interval': assistant.narration_interval})


if __name__ == '__main__':
    # Create static folder for audio files
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*50)
    print("EchoGuide Blind Navigation Assistant")
    print("="*50)
    print("📱 Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n Features:")
    print("   ✓ YOLOv8 Object Detection")
    print("   ✓ Text-to-Speech Narration")
    print("   ✓ Real-time Video Feed")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        assistant.release()
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

app = Flask(__name__, template_folder='mobile_app/frontend/templates')
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
        """Generate smart narration for navigation"""
        if not detections:
            return "Path appears clear"
        
        # Categorize detections
        people = [d for d in detections if d['class'] == 'person']
        vehicles = [d for d in detections if d['class'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]
        obstacles = [d for d in detections if d['class'] in ['chair', 'couch', 'table', 'dining table', 'bench', 'potted plant']]
        other = [d for d in detections if d not in people + vehicles + obstacles]
        
        narration_parts = []
        
        # 1. IMMEDIATE DANGERS (people/vehicles very close ahead)
        immediate_people = [p for p in people if 'very close' in p['position'] and 'ahead' in p['position']]
        if immediate_people:
            narration_parts.append("Person directly ahead, stop")
        
        immediate_vehicles = [v for v in vehicles if 'very close' in v['position'] or 'nearby' in v['position']]
        if immediate_vehicles:
            vehicle_type = immediate_vehicles[0]['class']
            position = immediate_vehicles[0]['position']
            narration_parts.append(f"Warning: {vehicle_type} {position}")
        
        # 2. OBSTACLES (most important for navigation)
        if obstacles:
            sorted_obstacles = sort_by_importance(obstacles, PRIORITY_MAP)
            closest_obstacle = sorted_obstacles[0]
            narration_parts.append(f"{closest_obstacle['class']} {closest_obstacle['position']}")
            
            # Mention second obstacle if it's in a different direction
            if len(sorted_obstacles) > 1:
                second = sorted_obstacles[1]
                first_pos = closest_obstacle['position']
                second_pos = second['position']
                
                # Only mention if in different direction
                if ('left' in first_pos and 'right' in second_pos) or \
                ('right' in first_pos and 'left' in second_pos) or \
                ('ahead' in first_pos and ('left' in second_pos or 'right' in second_pos)):
                    narration_parts.append(f"{second['class']} {second['position']}")
        
        # 3. PEOPLE SUMMARY (only if not immediate danger)
        if people and not immediate_people:
            # Count people by position
            people_left = sum(1 for p in people if 'left' in p['position'])
            people_right = sum(1 for p in people if 'right' in p['position'])
            people_ahead = sum(1 for p in people if 'ahead' in p['position'] and 'very close' not in p['position'])
            
            people_summary = []
            if people_ahead > 0:
                if people_ahead == 1:
                    people_summary.append("one person ahead")
                else:
                    people_summary.append(f"{people_ahead} people ahead")
            
            if people_left > 0:
                people_summary.append(f"{people_left} on left")
            
            if people_right > 0:
                people_summary.append(f"{people_right} on right")
            
            if people_summary:
                narration_parts.append(", ".join(people_summary))
        
        # 4. OTHER IMPORTANT OBJECTS (traffic lights, stop signs, etc.)
        priority_other = [o for o in other if o['class'] in ['traffic light', 'stop sign', 'fire hydrant']]
        if priority_other:
            obj = priority_other[0]
            narration_parts.append(f"{obj['class']} {obj['position']}")
        
        # Combine narration
        if not narration_parts:
            return "Area crowded with people"
        
        return ". ".join(narration_parts)
    
    
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
                # Generate audio in background thread to avoid blocking video
                threading.Thread(target=self.generate_audio, args=(self.current_narration,), daemon=True).start()
            
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
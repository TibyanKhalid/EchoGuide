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
from yolo_detection.scene_narrator import SceneNarrator
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
        
        # AUDIO STATE FIXES
        self.last_spoken_narration = None
        self.last_audio_path = None
        
        # Camera
        self.camera = cv2.VideoCapture(0)
        self.camera_lock = threading.Lock()

        self.narrator = SceneNarrator()
        self.previous_detections = []
        
        print("[Init] Ready!")
        
    def generate_narration(self, detections):
        """Generate intelligent scene narration"""
        return self.narrator.generate_scene_description(detections)

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
                
                # Get position description
                position_info = get_position_description(bbox, width, height)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox.tolist(),
                    'size': bbox_area,
                    'position': position_info  
                })
        
        return detections
    
    def generate_narration(self, detections):
        """Generate smart narration for navigation"""
        if not detections:
            return "Path appears clear"
        
        people = [d for d in detections if d['class'] == 'person']
        vehicles = [d for d in detections if d['class'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]
        obstacles = [d for d in detections if d['class'] in ['chair', 'couch', 'table', 'dining table', 'bench', 'potted plant']]
        other = [d for d in detections if d not in people + vehicles + obstacles]
        
        narration_parts = []
        
        # 1. Immediate danger
        immediate_people = [p for p in people if 'very close' in p['position']['description'] and 'ahead' in p['position']['description']]
        if immediate_people:
            narration_parts.append("Person directly ahead, stop")
        
        immediate_vehicles = [v for v in vehicles if 'very close' in v['position']['description'] or 'nearby' in v['position']['description']]
        if immediate_vehicles:
            vehicle_type = immediate_vehicles[0]['class']
            position = immediate_vehicles[0]['position']['description']
            narration_parts.append(f"Warning: {vehicle_type} {position}")
        
        # 2. Obstacles
        if obstacles:
            sorted_obstacles = sort_by_importance(obstacles, PRIORITY_MAP)
            closest_obstacle = sorted_obstacles[0]
            narration_parts.append(f"{closest_obstacle['class']} {closest_obstacle['position']['description']}")
            
            if len(sorted_obstacles) > 1:
                second = sorted_obstacles[1]
                if ('left' in closest_obstacle['position']['description'] and 'right' in second['position']['description']) or \
                ('right' in closest_obstacle['position']['description'] and 'left' in second['position']['description']) or \
                ('ahead' in closest_obstacle['position']['description'] and ('left' in second['position']['description'] or 'right' in second['position']['description'])):
                    narration_parts.append(f"{second['class']} {second['position']['description']}")
        
        # 3. People summary
        if people and not immediate_people:
            people_left = sum(1 for p in people if 'left' in p['position']['description'])
            people_right = sum(1 for p in people if 'right' in p['position']['description'])
            people_ahead = sum(1 for p in people if 'ahead' in p['position']['description'] and 'very close' not in p['position']['description'])
            
            people_summary = []
            if people_ahead > 0:
                people_summary.append(f"{people_ahead} ahead")
            if people_left > 0:
                people_summary.append(f"{people_left} left")
            if people_right > 0:
                people_summary.append(f"{people_right} right")
            
            if people_summary:
                narration_parts.append(", ".join(people_summary))
        
        priority_other = [o for o in other if o['class'] in ['traffic light', 'stop sign', 'fire hydrant']]
        if priority_other:
            obj = priority_other[0]
            narration_parts.append(f"{obj['class']} {obj['position']['description']}")
        
        if not narration_parts:
            return "Area crowded with people"
        
        return ". ".join(narration_parts)
    
    
    def generate_audio(self, text):
        """Generate smooth audio with unique filenames"""
        try:
            timestamp = int(time.time() * 1000)
            audio_path = f"static/narration_{timestamp}.mp3"

            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)

            self.last_audio_path = audio_path

            # Remove old audio files
            for f in os.listdir("static"):
                if f.startswith("narration_") and f != os.path.basename(audio_path):
                    try:
                        os.remove(os.path.join("static", f))
                    except:
                        pass

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
            
            # Narration timing
            current_time = time.time()
            if current_time - self.last_narration_time >= self.narration_interval:
                self.last_narration_time = current_time
                new_narration = self.generate_narration(self.current_detections)
                print(f"[NARRATION]: {new_narration}")
                
                # Only generate audio if narration changed
                if new_narration != self.last_spoken_narration:
                    self.last_spoken_narration = new_narration
                    self.current_narration = new_narration

                    # Generate audio in background thread to avoid blocking video
                    threading.Thread(target=self.generate_audio, args=(new_narration,), daemon=True).start()

            
            self.current_frame = frame
            return frame
    
    def release(self):
        self.camera.release()


assistant = WebNavigationAssistant()


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    while True:
        frame = assistant.get_frame()
        if frame is None:
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_narration')
def get_narration():
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
    if assistant.last_audio_path and os.path.exists(assistant.last_audio_path):
        return jsonify({'audio_url': '/' + assistant.last_audio_path})
    return jsonify({'audio_url': None})


@app.route('/set_interval', methods=['POST'])
def set_interval():
    data = request.get_json()
    interval = data.get('interval', 5)
    assistant.narration_interval = max(3, min(15, interval))
    print(f"[SETTING] Narration interval changed to {assistant.narration_interval}s")
    return jsonify({'success': True, 'interval': assistant.narration_interval})


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*50)
    print("EchoGuide Blind Navigation Assistant")
    print("="*50)
    print("📱 Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n Features:")
    print("   ✓ YOLOv8 Object Detection")
    print("   ✓ Text-to-Speech Narration (Smooth)")
    print("   ✓ Real-time Video Feed")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        assistant.release()
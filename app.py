# app.py
import os
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
from yolo_detection.llm_narrator import LLMNarrator

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
        
        # AUDIO STATE
        self.last_spoken_narration = None
        self.last_audio_path = None
        
        # Camera
        self.camera = cv2.VideoCapture(0)
        self.camera_lock = threading.Lock()

        # LLM Narrator
        print("[Init] Loading LLM for smart narration...")
        self.llm_narrator = LLMNarrator()
        self.use_llm = True  # Toggle to enable/disable
        
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
        """Generate narration with LLM or fallback"""
        if not detections:
            return "Path is clear. Continue forward."
        
        # Use LLM if enabled
        if self.use_llm:
            try:
                print("[LLM] Generating instruction...")
                instruction = self.llm_narrator.generate_navigation_instruction(detections)
                print(f"[LLM] Generated: {instruction}")
                return instruction
            except Exception as e:
                print(f"[LLM ERROR]: {e}")
                # Fall back to simple narration
                return self._generate_simple_narration(detections)
        else:
            return self._generate_simple_narration(detections)
    
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

                    # Generate audio in background thread
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
                'position': d['position']['description'],  # Extract description string
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
    print("EchoGuide: AI-Powered Audio Guide for the Visually Impaired")
    print("="*50)
    print("Open your browser and go to:")
    print("   http://localhost:5000")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        assistant.release()
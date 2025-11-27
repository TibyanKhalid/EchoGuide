# Import required libraries
import cv2
from ultralytics import YOLO
import time
from spatial_utils import get_position_description, sort_by_importance
from navigation_classes import NAVIGATION_CLASSES, PRIORITY_MAP
import pyttsx3
import threading
from queue import Queue
import threading
from gtts import gTTS
import os
import tempfile
from playsound import playsound

class NavigationAssistant:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.last_narration_time = 0
        self.narration_interval = 5
        
        # Speech queue for gTTS
        self.speech_queue = Queue()
        self.is_speaking = False
        
        # Start speech worker
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
    
    def _speech_worker(self):
        """Worker thread that processes speech using gTTS"""

        print("[WORKER] Speech worker started (using gTTS)")
        
        while True:
            text = self.speech_queue.get()
            print(f"[WORKER] Got text: {text}")
            
            if text:
                try:
                    self.is_speaking = True
                    print(f"[SPEAKING]: {text}")
                    
                    #Generate speech with gTTS
                    tts = gTTS(text=text, lang='en', slow=False)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                        temp_file = fp.name
                    
                    tts.save(temp_file)
                    
                    # Play the audio
                    playsound(temp_file)
                    
                    # Cleaning up
                    os.remove(temp_file)
                    
                    print(f"[DONE SPEAKING]")
                    self.is_speaking = False
                    
                except Exception as e:
                    print(f"[ERROR] Speech failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.is_speaking = False
    
    def speak(self, text):
        """Add text to speech queue"""
        # Clear old pending messages (only speak the latest one)
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break
        
        self.speech_queue.put(text)
        
    def process_frame(self, frame):
        """
        Process a single frame and return navigation-relevant detections
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        height, width = frame.shape[:2]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            # Filter for navigation-relevant classes
            if class_name in NAVIGATION_CLASSES:
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'size': bbox_area,
                    'position': get_position_description(bbox, width, height)
                })
        
        return detections
    
    def generate_narration_text(self, detections):
        """
        Convert detections to natural language description
        """
        if not detections:
            return "No obstacles detected"
        
        # Sort by importance
        sorted_dets = sort_by_importance(detections, PRIORITY_MAP)
        
        #Take top 3 most important objects
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
    
    def should_narrate(self):
        """
        Check if enough time has passed since last narration
        """
        current_time = time.time()
        if current_time - self.last_narration_time >= self.narration_interval:
            self.last_narration_time = current_time
            return True
        return False


def main():
    assistant = NavigationAssistant(conf_threshold=0.25)
    cap = cv2.VideoCapture(0)
    
    print("Navigation Assistant Started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detections = assistant.process_frame(frame)
        
        # Generate and display narration text
        if assistant.should_narrate():
            narration = assistant.generate_narration_text(detections)
            print(f"\n[NARRATION]: {narration}")
            assistant.speak(narration)
        
        # Draw bounding boxes for visualization
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']} ({det['confidence']:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Navigation Assistant', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# yolo_detection/llm_narrator.py

import os
import requests
import json

class LLMNarrator:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize LLM narrator using HuggingFace Inference API
        """
        print(f"[LLM] Initializing API connection to {model_name}...")
        
        # Get API token from environment
        self.api_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token not found! "
                "Set environment variable: HUGGINGFACE_TOKEN or HF_TOKEN"
            )
        
        # API endpoint
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Test connection
        self._test_connection()
        
        print("[LLM] API connection successful!")
    
    def _test_connection(self):
        """Test if the API is accessible"""
        try:
            test_payload = {
                "inputs": "Hello",
                "parameters": {"max_new_tokens": 10}
            }
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 503:
                print("[LLM] Model is loading... This may take 20-30 seconds on first use.")
                return
            
            if response.status_code != 200:
                print(f"[LLM WARNING] API returned status {response.status_code}")
                print(f"[LLM WARNING] Response: {response.text}")
            
        except Exception as e:
            print(f"[LLM WARNING] Connection test failed: {e}")
    
    def generate_navigation_instruction(self, detections):
        """
        Generate helpful navigation instruction from detections
        """
        if not detections:
            return "The path ahead is clear. You can continue walking forward."
        
        # Build detection summary
        detection_summary = self._build_detection_summary(detections)
        
        # Create prompt
        prompt = self._create_prompt(detection_summary)
        
        # Generate instruction
        instruction = self._generate(prompt)
        
        return instruction
    
    def _build_detection_summary(self, detections):
        """Convert detections to text summary"""
        summary_parts = []
        
        # Sort by urgency
        sorted_dets = sorted(
            detections,
            key=lambda d: (
                0 if d['position']['urgency'] == 'immediate' else
                1 if d['position']['urgency'] == 'caution' else
                2 if d['position']['urgency'] == 'aware' else 3
            )
        )
        
        # Take top 5 most important
        for det in sorted_dets[:5]:
            obj_class = det['class']
            position = det['position']
            urgency = position['urgency']
            horizontal = position['horizontal']
            distance = position['distance']
            
            summary_parts.append(
                f"{obj_class}: {distance} {horizontal} (urgency: {urgency})"
            )
        
        return "\n".join(summary_parts)
    
    def _create_prompt(self, detection_summary):
        """Create instruction prompt"""
        prompt = f"""You are a navigation assistant for blind people. Give ONE clear, actionable instruction based on these detected objects.

Detected objects:
{detection_summary}

Rules:
- Keep response under 20 words
- Start with most urgent obstacle
- Give specific directions: "move left", "move right", "stop", "continue forward"
- Be direct and helpful
- Format: [Action]. [Brief context if needed].

Instruction:"""
        
        return prompt
    
    def _generate(self, prompt):
        """Generate text using HuggingFace API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            # Handle different response codes
            if response.status_code == 503:
                return "System loading. Please wait a moment."
            
            if response.status_code == 429:
                return "Service busy. Proceeding with caution."
            
            if response.status_code != 200:
                print(f"[LLM API ERROR]: {response.status_code}")
                print(f"[LLM API RESPONSE]: {response.text}")
                return "Please proceed carefully."
            
            # Parse response
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '').strip()
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '').strip()
            else:
                generated_text = str(result).strip()
            
            # Clean up the response
            instruction = self._clean_instruction(generated_text)
            
            return instruction
            
        except requests.exceptions.Timeout:
            print("[LLM ERROR]: Request timeout")
            return "Navigation system is slow. Proceed with caution."
        
        except Exception as e:
            print(f"[LLM API ERROR]: {e}")
            return "Please proceed carefully."
    
    def _clean_instruction(self, text):
        """Clean up generated instruction"""
        # Remove common artifacts
        text = text.replace("Instruction:", "").strip()
        text = text.replace("Response:", "").strip()
        
        # Take first 1-2 sentences only
        sentences = text.split('.')
        if len(sentences) > 2:
            text = '. '.join(sentences[:2]) + '.'
        
        # Limit length
        words = text.split()
        if len(words) > 25:
            text = ' '.join(words[:25]) + '...'
        
        return text.strip()


# Test function
if __name__ == "__main__":
    print("Testing LLM Narrator...")
    
    narrator = LLMNarrator()
    
    # Test with sample detections
    test_detections = [
        {
            'class': 'chair',
            'position': {
                'description': 'very close directly ahead',
                'urgency': 'immediate',
                'horizontal': 'directly ahead',
                'distance': 'very close'
            }
        },
        {
            'class': 'person',
            'position': {
                'description': 'a few steps away on your left',
                'urgency': 'aware',
                'horizontal': 'on your left',
                'distance': 'a few steps away'
            }
        }
    ]
    
    instruction = narrator.generate_navigation_instruction(test_detections)
    print(f"\nGenerated Instruction: {instruction}")
# llm_narrator.py

import os
from groq import Groq
import requests

class LLMNarrator:
    def __init__(self):
        print("[LLM] Initializing Groq API...")
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment!")
        
        self.client = Groq(api_key=api_key)
        print("[LLM] Groq API ready!")
    
    def generate_navigation_instruction(self, detections):
        if not detections:
            return "Path is clear. Continue forward."
        
        detection_summary = self._build_detection_summary(detections)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Very fast
                messages=[
                    {
                        "role": "system",
                        "content": "You are a navigation assistant for blind people. Give ONE clear instruction under 20 words. Start with the action: Stop/Move left/Move right/Continue."
                    },
                    {
                        "role": "user",
                        "content": f"Objects detected:\n{detection_summary}\n\nNavigation instruction:"
                    }
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            instruction = response.choices[0].message.content.strip()
            return self._clean_instruction(instruction)
            
        except Exception as e:
            print(f"[LLM ERROR]: {e}")
            return "Proceed with caution."
    
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
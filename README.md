# EchoGuide: AI-Powered Audio Guide for the Visually Impaired
## **Overview**
This project enables real-time scene understanding and spoken navigation for visually impaired users using computer vision and AI.

## **Pipeline Archeticture**
Camera -> [YOLOv8] -> List of Objects -> [Florence-2] -> Text Description -> [text_to_speech] -> Audio Output

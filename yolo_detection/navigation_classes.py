NAVIGATION_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'bench': 13,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'dining table': 60,
    'tv': 62,
    'laptop': 63,
    'cell phone': 67,
    'book': 73,
    'clock': 74,
  #  'door': None,  # Not in COCO
  #  'stairs': None  # Not in COCO  
}

# Priority levels for narration (1 = highest priority)
PRIORITY_MAP = {
    'person': 1,
    'car': 1,
    'bus': 1,
    'truck': 1,
    'motorcycle': 1,
    'bicycle': 1,
    'traffic light': 1,
    'stop sign': 1,
  # 'stairs': 1,  
  # 'door': 2,    
    'fire hydrant': 2,
    'bench': 3,
    'chair': 3,
    'potted plant': 3,
}
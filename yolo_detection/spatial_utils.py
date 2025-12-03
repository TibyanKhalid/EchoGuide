# spatial_utils.py

def get_position_description(bbox, frame_width, frame_height):
    """
    Convert bounding box to natural, detailed position description
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Granular horizontal position 
    if center_x < frame_width * 0.2:
        horizontal = "on your far left"
    elif center_x < frame_width * 0.4:
        horizontal = "on your left"
    elif center_x < frame_width * 0.6:
        horizontal = "directly ahead"
    elif center_x < frame_width * 0.8:
        horizontal = "on your right"
    else:
        horizontal = "on your far right"
    
    # Vertical position (helps with stairs, overhead objects)
    if center_y < frame_height * 0.3:
        vertical = "above"
    elif center_y > frame_height * 0.7:
        vertical = "at ground level"
    else:
        vertical = None  # Eye level, don't mention
    
    # Better depth estimation
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    relative_size = bbox_area / frame_area
    
    if relative_size > 0.35:
        distance = "right in front of you"
        urgency = "immediate"
    elif relative_size > 0.2:
        distance = "very close"
        urgency = "immediate"
    elif relative_size > 0.1:
        distance = "nearby"
        urgency = "caution"
    elif relative_size > 0.04:
        distance = "a few steps away"
        urgency = "aware"
    else:
        distance = "in the distance"
        urgency = "info"
    
    # Combine position
    if vertical:
        position = f"{distance} {horizontal}, {vertical}"
    else:
        position = f"{distance} {horizontal}"
    
    return {
        'description': position,
        'urgency': urgency,
        'horizontal': horizontal,
        'distance': distance
    }


def sort_by_importance(detections, priority_map):
    """
    Sort detections by urgency and priority
    """
    def get_sort_key(det):
        class_name = det['class']
        priority = priority_map.get(class_name, 5)
        
        # Urgency from position
        urgency_score = {
            'immediate': 0,
            'caution': 1,
            'aware': 2,
            'info': 3
        }
        urgency = urgency_score.get(det['position']['urgency'], 3)
        
        size = det['size']
        
        # Sort by: urgency first, then priority, then size
        return (urgency, priority, -size)
    
    return sorted(detections, key=get_sort_key)


def get_direction_advice(position_desc):
    """
    Convert position to actionable direction
    """
    horizontal = position_desc['horizontal']
    urgency = position_desc['urgency']
    
    if urgency == 'immediate':
        if 'ahead' in horizontal:
            return "stop"
        elif 'left' in horizontal:
            return "move right"
        elif 'right' in horizontal:
            return "move left"
    
    return None  # No immediate action needed
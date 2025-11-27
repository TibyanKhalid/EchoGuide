def get_position_description(bbox, frame_width, frame_height):
    """
    Convert bounding box to natural language position
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    #Horizontal position
    if center_x < frame_width * 0.33:
        horizontal = "on your left"
    elif center_x < frame_width * 0.67:
        horizontal = "ahead"
    else:
        horizontal = "on your right"
    
    # Depth estimation (rough and based on bbox size)
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    relative_size = bbox_area / frame_area
    
    if relative_size > 0.3:
        distance = "very close"
    elif relative_size > 0.15:
        distance = "nearby"
    elif relative_size > 0.05:
        distance = "at medium distance"
    else:
        distance = "far away"
    
    return f"{distance} {horizontal}"


def sort_by_importance(detections, priority_map):
    """
    Sort detections by priority and proximity
    """
    def get_priority(det):
        class_name = det['class']
        priority = priority_map.get(class_name, 5)
        size = det['size']  # Larger = closer = more important
        return (priority, -size)  # Sort by priority first, then size descending
    
    return sorted(detections, key=get_priority)
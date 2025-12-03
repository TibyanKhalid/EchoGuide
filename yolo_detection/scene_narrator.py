# scene_narrator.py

class SceneNarrator:
    """
    Generates intelligent, context-aware scene descriptions
    """
    
    @staticmethod
    def generate_scene_description(detections):
        """
        Create a natural, helpful narration from detections
        """
        if not detections:
            return "Path is clear, continue forward"
        
        # Categorize detections
        categories = SceneNarrator._categorize_detections(detections)
        
        # Build narration parts
        narration = []
        
        # 1. IMMEDIATE ALERTS (urgent obstacles)
        immediate = SceneNarrator._handle_immediate_alerts(categories)
        if immediate:
            narration.append(immediate)
        
        # 2. NAVIGATION OBSTACLES
        obstacles = SceneNarrator._describe_obstacles(categories)
        if obstacles:
            narration.append(obstacles)
        
        # 3. ENVIRONMENT DESCRIPTION
        environment = SceneNarrator._describe_environment(categories)
        if environment:
            narration.append(environment)
        
        # 4. SAFE PATH GUIDANCE
        guidance = SceneNarrator._suggest_path(categories)
        if guidance:
            narration.append(guidance)
        
        if not narration:
            return "Multiple objects detected, proceed with caution"
        
        return ". ".join(narration)
    
    @staticmethod
    def _categorize_detections(detections):
        """Group detections by type and urgency"""
        return {
            'immediate_danger': [d for d in detections if d['position']['urgency'] == 'immediate'],
            'people': [d for d in detections if d['class'] == 'person'],
            'vehicles': [d for d in detections if d['class'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']],
            'furniture': [d for d in detections if d['class'] in ['chair', 'couch', 'table', 'dining table', 'bench']],
            'obstacles': [d for d in detections if d['class'] in ['potted plant', 'suitcase', 'backpack', 'umbrella']],
            'landmarks': [d for d in detections if d['class'] in ['traffic light', 'stop sign', 'fire hydrant', 'clock']],
            'electronics': [d for d in detections if d['class'] in ['tv', 'laptop', 'cell phone']],
        }
    
    @staticmethod
    def _handle_immediate_alerts(categories):
        """Handle urgent, immediate obstacles"""
        immediate = categories['immediate_danger']
        if not immediate:
            return None
        
        alerts = []
        for obj in immediate[:2]:  # Max 2 immediate alerts
            obj_class = obj['class']
            position = obj['position']['description']
            
            # Get action advice
            from spatial_utils import get_direction_advice
            action = get_direction_advice(obj['position'])
            
            if action:
                alerts.append(f"{obj_class} {position}, {action}")
            else:
                alerts.append(f"Warning: {obj_class} {position}")
        
        return ". ".join(alerts)
    
    @staticmethod
    def _describe_obstacles(categories):
        """Describe physical obstacles to navigate around"""
        furniture = categories['furniture']
        obstacles = categories['obstacles']
        
        all_obstacles = furniture + obstacles
        if not all_obstacles:
            return None
        
        # Group by direction
        left_obs = [o for o in all_obstacles if 'left' in o['position']['horizontal']]
        right_obs = [o for o in all_obstacles if 'right' in o['position']['horizontal']]
        center_obs = [o for o in all_obstacles if 'ahead' in o['position']['horizontal']]
        
        descriptions = []
        
        if center_obs:
            closest = center_obs[0]
            descriptions.append(f"{closest['class']} blocking path {closest['position']['distance']}")
        
        if left_obs and right_obs:
            descriptions.append(f"obstacles on both sides")
        elif left_obs:
            descriptions.append(f"{left_obs[0]['class']} on left side")
        elif right_obs:
            descriptions.append(f"{right_obs[0]['class']} on right side")
        
        return ", ".join(descriptions) if descriptions else None
    
    @staticmethod
    def _describe_environment(categories):
        """Describe the general environment"""
        people = categories['people']
        vehicles = categories['vehicles']
        landmarks = categories['landmarks']
        
        descriptions = []
        
        # People summary
        if people:
            count = len(people)
            if count == 1:
                person = people[0]
                if person['position']['urgency'] != 'immediate':
                    descriptions.append(f"one person {person['position']['horizontal']}")
            elif count <= 3:
                descriptions.append(f"{count} people nearby")
            else:
                descriptions.append(f"crowded area with {count} people")
        
        # Vehicles
        if vehicles:
            vehicle = vehicles[0]
            descriptions.append(f"{vehicle['class']} {vehicle['position']['horizontal']}")
        
        # Landmarks for orientation
        if landmarks:
            landmark = landmarks[0]
            descriptions.append(f"{landmark['class']} {landmark['position']['horizontal']}")
        
        return ", ".join(descriptions) if descriptions else None
    
    @staticmethod
    def _suggest_path(categories):
        """Suggest safe navigation path"""
        immediate = categories['immediate_danger']
        furniture = categories['furniture']
        
        # If immediate danger, path suggestion already given
        if immediate:
            return None
        
        # Check if path is blocked
        center_blocked = any(
            'ahead' in obj['position']['horizontal'] 
            for obj in furniture + categories['obstacles']
        )
        
        left_clear = not any(
            'left' in obj['position']['horizontal'] and obj['position']['urgency'] in ['immediate', 'caution']
            for obj in furniture + categories['obstacles']
        )
        
        right_clear = not any(
            'right' in obj['position']['horizontal'] and obj['position']['urgency'] in ['immediate', 'caution']
            for obj in furniture + categories['obstacles']
        )
        
        if center_blocked:
            if left_clear and right_clear:
                return "path ahead blocked, space available on both sides"
            elif left_clear:
                return "clear path on your left"
            elif right_clear:
                return "clear path on your right"
        
        return None
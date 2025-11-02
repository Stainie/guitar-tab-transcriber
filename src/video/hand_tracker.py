"""
Hand tracking using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional


class HandTracker:
    """Track hands and finger positions using MediaPipe"""
    
    def __init__(self, 
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 max_num_hands=2):
        """
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            max_num_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hands in frame
        
        Args:
            frame: RGB image frame
            
        Returns:
            List of detected hands with landmarks
        """
        # Process frame
        results = self.hands.process(frame)
        
        detected_hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Determine if left or right hand
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                hand_score = handedness.classification[0].score
                
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,  # Relative depth
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                detected_hands.append({
                    'label': hand_label,
                    'score': hand_score,
                    'landmarks': landmarks,
                    'raw_landmarks': hand_landmarks  # Keep for drawing
                })
        
        return detected_hands
    
    def get_fingertip_positions(self, hand_data: Dict) -> Dict[str, tuple]:
        """
        Extract fingertip positions from hand landmarks
        
        MediaPipe hand landmark indices:
        - 4: Thumb tip
        - 8: Index finger tip
        - 12: Middle finger tip
        - 16: Ring finger tip
        - 20: Pinky tip
        
        Args:
            hand_data: Hand data from detect_hands()
            
        Returns:
            Dictionary of finger names to (x, y, z) coordinates
        """
        landmarks = hand_data['landmarks']
        
        fingertips = {
            'thumb': (landmarks[4]['x'], landmarks[4]['y'], landmarks[4]['z']),
            'index': (landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']),
            'middle': (landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']),
            'ring': (landmarks[16]['x'], landmarks[16]['y'], landmarks[16]['z']),
            'pinky': (landmarks[20]['x'], landmarks[20]['y'], landmarks[20]['z'])
        }
        
        return fingertips
    
    def get_wrist_position(self, hand_data: Dict) -> tuple:
        """Get wrist position (landmark 0)"""
        landmarks = hand_data['landmarks']
        return (landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z'])
    
    def draw_hands_on_frame(self, frame: np.ndarray, hands: List[Dict]) -> np.ndarray:
        """
        Draw hand landmarks on frame for visualization
        
        Args:
            frame: RGB image frame
            hands: List of detected hands
            
        Returns:
            Frame with hand landmarks drawn
        """
        annotated_frame = frame.copy()
        
        for hand in hands:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand['raw_landmarks'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Add label
            wrist = hand['landmarks'][0]
            h, w, _ = annotated_frame.shape
            label_pos = (int(wrist['x'] * w), int(wrist['y'] * h) - 20)
            
            cv2.putText(
                annotated_frame,
                f"{hand['label']} ({hand['score']:.2f})",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return annotated_frame
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()
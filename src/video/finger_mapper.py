"""
Map finger positions to guitar strings and frets - Phase 2.3
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from config.guitar_config import NUM_STRINGS, NUM_FRETS


class FingerMapper:
    """Map detected finger positions to guitar strings and frets"""
    
    def __init__(self):
        self.fretboard_region = None
        self.string_lines = None
        self.fret_positions = None
    
    def calibrate(self, fretboard_info: dict, string_lines: List[Tuple], frame_shape: Tuple):
        """
        Calibrate the mapper with detected fretboard and strings
        
        Args:
            fretboard_info: Fretboard detection info with bbox
            string_lines: List of detected string lines
            frame_shape: Shape of the frame (height, width, channels)
        """
        if not fretboard_info or not string_lines:
            return False
        
        self.fretboard_region = fretboard_info['bbox']
        self.string_lines = string_lines
        
        # Calculate fret positions (approximate)
        x, y, w, h = self.fretboard_region
        
        # Frets get closer together as you go up the neck
        # Use exponential spacing to approximate real fret positions
        self.fret_positions = []
        
        # Calculate fret spacing (12th root of 2 for equal temperament)
        fret_constant = 2 ** (1/12)
        
        for fret_num in range(NUM_FRETS + 1):
            # Distance from nut to fret
            # Scale length * (1 - (1 / 2^(fret/12)))
            ratio = 1 - (1 / (fret_constant ** fret_num))
            fret_x = x + int(w * ratio * 0.85)  # 0.85 because we don't see full neck
            
            self.fret_positions.append(fret_x)
        
        return True
    
    def map_hand_to_fretboard(self, hand_data: Dict, frame_shape: Tuple) -> Optional[Dict]:
        """
        Map a detected hand to fretboard positions
        
        Args:
            hand_data: Hand detection data with landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Dictionary with finger-to-string/fret mappings or None
        """
        if not self.is_calibrated():
            return None
        
        h, w = frame_shape[:2]
        
        # Determine if this is fretting hand (left) or picking hand (right)
        # For standard right-handed playing:
        # - Left hand (labeled "Right" by MediaPipe due to camera flip) is on fretboard
        # - Right hand (labeled "Left" by MediaPipe) is picking/strumming
        
        hand_label = hand_data['label']
        
        # Check if hand is over fretboard
        wrist = hand_data['landmarks'][0]
        wrist_x = int(wrist['x'] * w)
        wrist_y = int(wrist['y'] * h)
        
        if not self._is_point_near_fretboard(wrist_x, wrist_y):
            return None
        
        # Get fingertip positions in pixel coordinates
        landmarks = hand_data['landmarks']
        
        fingertips_pixel = {
            'thumb': (int(landmarks[4]['x'] * w), int(landmarks[4]['y'] * h)),
            'index': (int(landmarks[8]['x'] * w), int(landmarks[8]['y'] * h)),
            'middle': (int(landmarks[12]['x'] * w), int(landmarks[12]['y'] * h)),
            'ring': (int(landmarks[16]['x'] * w), int(landmarks[16]['y'] * h)),
            'pinky': (int(landmarks[20]['x'] * w), int(landmarks[20]['y'] * h))
        }
        
        # Get finger base positions (for detecting pressed vs lifted fingers)
        finger_bases = {
            'thumb': (int(landmarks[2]['x'] * w), int(landmarks[2]['y'] * h)),
            'index': (int(landmarks[5]['x'] * w), int(landmarks[5]['y'] * h)),
            'middle': (int(landmarks[9]['x'] * w), int(landmarks[9]['y'] * h)),
            'ring': (int(landmarks[13]['x'] * w), int(landmarks[13]['y'] * h)),
            'pinky': (int(landmarks[17]['x'] * w), int(landmarks[17]['y'] * h))
        }
        
        # Map each finger to string and fret
        finger_mappings = {}
        
        for finger_name, tip_pos in fingertips_pixel.items():
            base_pos = finger_bases[finger_name]
            
            # Check if finger is pressing (tip is closer to fretboard than base)
            is_pressing = self._is_finger_pressing(tip_pos, base_pos, hand_label)
            
            if is_pressing:
                string_idx = self._find_closest_string(tip_pos)
                fret_num = self._find_closest_fret(tip_pos)
                
                if string_idx is not None and fret_num is not None:
                    finger_mappings[finger_name] = {
                        'string': string_idx,
                        'fret': fret_num,
                        'position': tip_pos,
                        'confidence': hand_data['score']
                    }
        
        if not finger_mappings:
            return None
        
        return {
            'hand_label': hand_label,
            'finger_mappings': finger_mappings,
            'hand_type': 'fretting' if self._is_fretting_hand(hand_label, wrist_x) else 'picking'
        }
    
    def _is_point_near_fretboard(self, x: int, y: int, tolerance: int = 50) -> bool:
        """Check if a point is near the fretboard region"""
        if not self.fretboard_region:
            return False
        
        fb_x, fb_y, fb_w, fb_h = self.fretboard_region
        
        return (fb_x - tolerance <= x <= fb_x + fb_w + tolerance and
                fb_y - tolerance <= y <= fb_y + fb_h + tolerance)
    
    def _is_finger_pressing(self, tip_pos: Tuple, base_pos: Tuple, hand_label: str) -> bool:
        """
        Determine if a finger is pressing down on the fretboard
        
        For fretting hand, tip should be lower (higher y value) than base
        """
        tip_x, tip_y = tip_pos
        base_x, base_y = base_pos
        
        # Check if fingertip is on fretboard
        if not self._is_point_near_fretboard(tip_x, tip_y, tolerance=30):
            return False
        
        # For pressing, fingertip should be extended and closer to strings
        # This is a simplified heuristic
        distance = np.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
        
        # Finger should be reasonably extended
        return distance > 40  # pixels
    
    def _is_fretting_hand(self, hand_label: str, wrist_x: int) -> bool:
        """
        Determine if this is the fretting hand
        
        For right-handed players:
        - Fretting hand (left hand) is usually on left side of frame
        - But MediaPipe labels it as "Right" due to mirror effect
        
        For now, we'll use position: hand closer to left edge is fretting hand
        """
        if not self.fretboard_region:
            return False
        
        fb_x, fb_y, fb_w, fb_h = self.fretboard_region
        fb_center_x = fb_x + fb_w / 2
        
        # Hand on the left side of fretboard is typically fretting hand
        return wrist_x < fb_center_x
    
    def _find_closest_string(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Find the closest string to a given position
        
        Args:
            position: (x, y) position in pixels
            
        Returns:
            String index (0-5) or None
        """
        if not self.string_lines:
            return None
        
        x, y = position
        min_distance = float('inf')
        closest_string = None
        
        for string_idx, string_line in enumerate(self.string_lines):
            x1, y1, x2, y2 = string_line
            
            # Calculate distance from point to line segment
            distance = self._point_to_line_distance(x, y, x1, y1, x2, y2)
            
            if distance < min_distance:
                min_distance = distance
                closest_string = string_idx
        
        # Only return if distance is reasonable (within 30 pixels)
        if min_distance < 30:
            return closest_string
        
        return None
    
    def _find_closest_fret(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Find the closest fret to a given position
        
        Args:
            position: (x, y) position in pixels
            
        Returns:
            Fret number (0-NUM_FRETS) or None
        """
        if not self.fret_positions:
            return None
        
        x, y = position
        min_distance = float('inf')
        closest_fret = None
        
        for fret_num, fret_x in enumerate(self.fret_positions):
            distance = abs(x - fret_x)
            
            if distance < min_distance:
                min_distance = distance
                closest_fret = fret_num
        
        # Adjust for "between frets" positions
        # If finger is between two frets, choose the one closer to nut (lower fret)
        if closest_fret and closest_fret > 0:
            # Check if we're past this fret (towards bridge)
            if x > self.fret_positions[closest_fret]:
                # Finger is between this fret and the next, so we're pressing this fret
                pass
            else:
                # Finger is between previous fret and this one
                if closest_fret > 0:
                    closest_fret -= 1
        
        # Ensure fret is within valid range
        if closest_fret is not None and 0 <= closest_fret <= NUM_FRETS:
            return closest_fret
        
        return None
    
    def _point_to_line_distance(self, px: int, py: int, 
                                 x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Calculate distance from point to line segment
        """
        # Vector from line start to point
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line segment is a point
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t of projection of point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def is_calibrated(self) -> bool:
        """Check if mapper is calibrated"""
        return (self.fretboard_region is not None and 
                self.string_lines is not None and 
                self.fret_positions is not None)
    
    def get_played_notes(self, hand_mappings: List[Dict]) -> List[Dict]:
        """
        Determine which notes are being played based on hand mappings
        
        Args:
            hand_mappings: List of hand mapping dictionaries
            
        Returns:
            List of note dictionaries with string and fret info
        """
        played_notes = []
        
        # Find fretting hand
        fretting_hand = None
        for hand_mapping in hand_mappings:
            if hand_mapping and hand_mapping.get('hand_type') == 'fretting':
                fretting_hand = hand_mapping
                break
        
        if not fretting_hand:
            return played_notes
        
        # Get all pressed positions
        finger_mappings = fretting_hand['finger_mappings']
        
        # Group by string (in case multiple fingers on same string, take highest fret)
        string_frets = {}
        
        for finger_name, mapping in finger_mappings.items():
            string_idx = mapping['string']
            fret_num = mapping['fret']
            
            if string_idx not in string_frets or fret_num > string_frets[string_idx]['fret']:
                string_frets[string_idx] = {
                    'fret': fret_num,
                    'finger': finger_name,
                    'confidence': mapping['confidence']
                }
        
        # Convert to note list
        for string_idx, info in string_frets.items():
            played_notes.append({
                'string': string_idx,
                'fret': info['fret'],
                'finger': info['finger'],
                'confidence': info['confidence']
            })
        
        return played_notes
    
    def draw_finger_mappings(self, frame: np.ndarray, hand_mappings: List[Dict]) -> np.ndarray:
        """Draw finger-to-fretboard mappings on frame"""
        annotated = frame.copy()
        
        for hand_mapping in hand_mappings:
            if not hand_mapping:
                continue
            
            hand_type = hand_mapping.get('hand_type', 'unknown')
            finger_mappings = hand_mapping.get('finger_mappings', {})
            
            # Choose color based on hand type
            color = (0, 255, 0) if hand_type == 'fretting' else (255, 165, 0)
            
            for finger_name, mapping in finger_mappings.items():
                position = mapping['position']
                string_idx = mapping['string']
                fret_num = mapping['fret']
                
                # Draw circle at finger position
                cv2.circle(annotated, position, 8, color, -1)
                
                # Draw label
                label = f"{finger_name[0].upper()}: S{string_idx+1}F{fret_num}"
                cv2.putText(
                    annotated,
                    label,
                    (position[0] + 15, position[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
        
        return annotated
    
    def draw_fret_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw approximate fret positions for visualization"""
        if not self.is_calibrated():
            return frame
        
        annotated = frame.copy()
        
        # Draw fret lines
        for fret_num, fret_x in enumerate(self.fret_positions):
            if fret_num % 3 == 0:  # Draw every 3rd fret for clarity
                fb_x, fb_y, fb_w, fb_h = self.fretboard_region
                cv2.line(
                    annotated,
                    (fret_x, fb_y),
                    (fret_x, fb_y + fb_h),
                    (100, 100, 100),
                    1
                )
                
                # Label fret number
                if fret_num > 0:
                    cv2.putText(
                        annotated,
                        str(fret_num),
                        (fret_x - 5, fb_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (100, 100, 100),
                        1
                    )
        
        return annotated


# Import cv2 here since we use it in draw methods
import cv2
"""
Detect which strings are being played - Phase 2.4
"""
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import deque
from config.guitar_config import NUM_STRINGS


class StringStateDetector:
    """Detect which strings are actively being played"""
    
    def __init__(self, history_length=5):
        """
        Args:
            history_length: Number of frames to track for motion detection
        """
        self.history_length = history_length
        self.picking_hand_history = deque(maxlen=history_length)
        self.string_activation_history = deque(maxlen=history_length)
        
        # State tracking
        self.last_pick_time = {}  # Track when each string was last picked
        self.pick_threshold = 15  # Pixel movement threshold for pick detection
    
    def detect_string_activity(self, 
                               hands: List[Dict],
                               hand_mappings: List[Dict],
                               string_lines: List[Tuple],
                               frame_shape: Tuple,
                               timestamp: float) -> Dict:
        """
        Detect which strings are currently being played
        
        Args:
            hands: Detected hands from hand tracker
            hand_mappings: Finger-to-fretboard mappings
            string_lines: Detected string line positions
            frame_shape: Shape of frame (height, width, channels)
            timestamp: Current frame timestamp
            
        Returns:
            Dictionary with string activity information
        """
        h, w = frame_shape[:2]
        
        # Find picking hand
        picking_hand = self._find_picking_hand(hands, hand_mappings)
        
        if not picking_hand:
            return {
                'active_strings': [],
                'picking_detected': False,
                'strumming': False,
                'pick_direction': None
            }
        
        # Get picking hand position
        picking_position = self._get_picking_position(picking_hand, w, h)
        
        # Track picking hand motion
        self.picking_hand_history.append({
            'position': picking_position,
            'timestamp': timestamp
        })
        
        # Detect picking motion
        pick_motion = self._detect_pick_motion()
        
        # Detect which strings are affected
        active_strings = []
        strumming = False
        
        if pick_motion:
            # Determine which strings are in the pick path
            if string_lines:
                active_strings, strumming = self._find_affected_strings(
                    picking_position,
                    pick_motion,
                    string_lines,
                    timestamp
                )
        
        return {
            'active_strings': active_strings,
            'picking_detected': pick_motion is not None,
            'strumming': strumming,
            'pick_direction': pick_motion['direction'] if pick_motion else None,
            'picking_position': picking_position
        }
    
    def _find_picking_hand(self, hands: List[Dict], hand_mappings: List[Dict]) -> Optional[Dict]:
        """Find the picking hand (not the fretting hand)"""
        if not hands:
            return None
        
        # If we have hand mappings, find the non-fretting hand
        fretting_hands = set()
        for mapping in hand_mappings:
            if mapping and mapping.get('hand_type') == 'fretting':
                fretting_hands.add(mapping['hand_label'])
        
        # Return first hand that's not fretting
        for hand in hands:
            if hand['label'] not in fretting_hands:
                return hand
        
        # If no clear distinction, use position heuristic
        # Picking hand is usually on the right side for right-handed players
        if len(hands) == 1:
            return hands[0]
        
        # Choose hand with rightmost wrist position
        rightmost_hand = max(hands, key=lambda h: h['landmarks'][0]['x'])
        return rightmost_hand
    
    def _get_picking_position(self, hand: Dict, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """
        Get the effective picking position (index fingertip or pick position)
        """
        landmarks = hand['landmarks']
        
        # Use index fingertip (landmark 8) as picking point
        # In actual playing, this is close to where pick makes contact
        index_tip = landmarks[8]
        
        x = int(index_tip['x'] * frame_w)
        y = int(index_tip['y'] * frame_h)
        
        return (x, y)
    
    def _detect_pick_motion(self) -> Optional[Dict]:
        """
        Detect picking motion from hand position history
        
        Returns:
            Dictionary with motion info or None
        """
        if len(self.picking_hand_history) < 3:
            return None
        
        # Get recent positions
        positions = [h['position'] for h in self.picking_hand_history]
        timestamps = [h['timestamp'] for h in self.picking_hand_history]
        
        # Calculate velocity (recent movement)
        current_pos = positions[-1]
        prev_pos = positions[-2]
        
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Check if motion exceeds threshold
        if velocity < self.pick_threshold:
            return None
        
        # Determine pick direction
        direction = self._classify_pick_direction(dx, dy)
        
        # Time delta
        dt = timestamps[-1] - timestamps[-2]
        
        return {
            'velocity': velocity,
            'direction': direction,
            'dx': dx,
            'dy': dy,
            'dt': dt
        }
    
    def _classify_pick_direction(self, dx: float, dy: float) -> str:
        """Classify pick direction based on motion vector"""
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Normalize to 0-360
        if angle < 0:
            angle += 360
        
        # Classify into directions
        if 45 <= angle < 135:
            return 'downstroke'  # Moving down
        elif 225 <= angle < 315:
            return 'upstroke'    # Moving up
        elif angle < 45 or angle >= 315:
            return 'across-right'
        else:
            return 'across-left'
    
    def _find_affected_strings(self,
                               picking_position: Tuple[int, int],
                               pick_motion: Dict,
                               string_lines: List[Tuple],
                               timestamp: float) -> Tuple[List[int], bool]:
        """
        Determine which strings are affected by the picking motion
        
        Returns:
            Tuple of (list of affected string indices, is_strumming)
        """
        if not string_lines:
            return [], False
        
        pick_x, pick_y = picking_position
        velocity = pick_motion['velocity']
        direction = pick_motion['direction']
        
        # Find strings near the pick position
        affected_strings = []
        min_distance_threshold = 40  # pixels
        
        for string_idx, string_line in enumerate(string_lines):
            x1, y1, x2, y2 = string_line
            
            # Calculate distance from pick to string
            distance = self._point_to_line_distance(pick_x, pick_y, x1, y1, x2, y2)
            
            if distance < min_distance_threshold:
                # Check if pick crossed the string recently
                if self._did_pick_cross_string(picking_position, string_line, pick_motion):
                    affected_strings.append(string_idx)
                    self.last_pick_time[string_idx] = timestamp
        
        # Determine if strumming (multiple strings in quick succession)
        is_strumming = len(affected_strings) > 1
        
        # If high velocity across motion, likely strumming even if only 1 string detected
        if velocity > 30 and direction in ['across-right', 'across-left', 'downstroke', 'upstroke']:
            is_strumming = True
            # Add nearby strings that might have been missed
            if affected_strings:
                primary_string = affected_strings[0]
                # Add adjacent strings
                for offset in [-1, 1]:
                    adjacent = primary_string + offset
                    if 0 <= adjacent < len(string_lines) and adjacent not in affected_strings:
                        affected_strings.append(adjacent)
        
        return sorted(affected_strings), is_strumming
    
    def _did_pick_cross_string(self, 
                               current_pos: Tuple[int, int],
                               string_line: Tuple,
                               pick_motion: Dict) -> bool:
        """
        Check if pick motion crossed a string
        """
        # For now, use simple proximity and motion direction
        # A more sophisticated version would track intersection of motion path with string
        
        x1, y1, x2, y2 = string_line
        pick_x, pick_y = current_pos
        
        # Check if pick is crossing perpendicular to string
        direction = pick_motion['direction']
        
        # Strings are mostly horizontal, so perpendicular crossing is vertical motion
        if direction in ['downstroke', 'upstroke']:
            return True
        
        # Or fast horizontal motion near a string
        if direction in ['across-right', 'across-left'] and pick_motion['velocity'] > 25:
            return True
        
        return False
    
    def _point_to_line_distance(self, px: int, py: int,
                                 x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate distance from point to line segment"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def combine_with_fretting(self, 
                             string_activity: Dict,
                             played_notes: List[Dict]) -> List[Dict]:
        """
        Combine string activity detection with fretting positions
        
        Args:
            string_activity: Result from detect_string_activity
            played_notes: Notes from finger mapper
            
        Returns:
            List of complete note events with string, fret, and timing
        """
        active_strings = string_activity['active_strings']
        
        if not active_strings:
            return []
        
        # Create complete note events
        note_events = []
        
        # Map fretting positions to active strings
        fretted_strings = {note['string']: note for note in played_notes}
        
        for string_idx in active_strings:
            if string_idx in fretted_strings:
                # String is being fretted
                note = fretted_strings[string_idx].copy()
                note['played'] = True
                note['strumming'] = string_activity['strumming']
                note_events.append(note)
            else:
                # Open string being played
                note_events.append({
                    'string': string_idx,
                    'fret': 0,  # Open string
                    'finger': None,
                    'played': True,
                    'strumming': string_activity['strumming'],
                    'confidence': 0.8
                })
        
        return note_events
    
    def draw_string_activity(self, 
                            frame: np.ndarray,
                            string_activity: Dict,
                            string_lines: List[Tuple]) -> np.ndarray:
        """Draw string activity visualization"""
        annotated = frame.copy()
        
        if not string_activity or not string_lines:
            return annotated
        
        active_strings = string_activity['active_strings']
        picking_detected = string_activity['picking_detected']
        
        # Highlight active strings
        for string_idx in active_strings:
            if string_idx < len(string_lines):
                x1, y1, x2, y2 = string_lines[string_idx]
                # Draw thick line for active strings
                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4)
        
        # Draw picking position
        if picking_detected and 'picking_position' in string_activity:
            pick_pos = string_activity['picking_position']
            cv2.circle(annotated, pick_pos, 10, (255, 0, 255), -1)
            
            # Draw pick direction indicator
            direction = string_activity.get('pick_direction')
            if direction:
                # Arrow indicating pick direction
                arrow_length = 30
                if direction == 'downstroke':
                    end_pos = (pick_pos[0], pick_pos[1] + arrow_length)
                elif direction == 'upstroke':
                    end_pos = (pick_pos[0], pick_pos[1] - arrow_length)
                elif direction == 'across-right':
                    end_pos = (pick_pos[0] + arrow_length, pick_pos[1])
                else:  # across-left
                    end_pos = (pick_pos[0] - arrow_length, pick_pos[1])
                
                cv2.arrowedLine(annotated, pick_pos, end_pos, (255, 0, 255), 3, tipLength=0.3)
        
        # Add text overlay
        if active_strings:
            text = f"Strumming" if string_activity['strumming'] else f"Picking"
            text += f": Strings {[s+1 for s in active_strings]}"
            
            cv2.putText(
                annotated,
                text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        return annotated
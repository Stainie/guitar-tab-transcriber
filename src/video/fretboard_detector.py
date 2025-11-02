"""
Fretboard detection and calibration - Phase 2.2
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from config.guitar_config import (
    FRETBOARD_MIN_AREA_RATIO,
    FRETBOARD_MAX_AREA_RATIO,
    FRETBOARD_ASPECT_RATIO_MIN,
    FRETBOARD_ASPECT_RATIO_MAX,
    STRING_DETECTION_SENSITIVITY,
    STRING_MIN_LENGTH_RATIO,
    CALIBRATION_FRAMES_REQUIRED,
    CALIBRATION_CONSISTENCY_THRESHOLD,
    NUM_STRINGS
)


class FretboardDetector:
    """Detect and track guitar fretboard in video"""
    
    def __init__(self, enable_calibration=True):
        """
        Args:
            enable_calibration: Whether to use multi-frame calibration
        """
        self.enable_calibration = enable_calibration
        self.is_calibrated = False
        
        # Calibration state
        self.calibration_history = deque(maxlen=CALIBRATION_FRAMES_REQUIRED)
        self.calibrated_region = None
        self.calibrated_strings = None
        
        # Tracking state
        self.last_fretboard = None
        self.detection_failures = 0
    
    def detect_fretboard(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect fretboard region in frame
        
        Args:
            frame: RGB image frame
            
        Returns:
            Dictionary with fretboard info or None if not detected
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        
        # If calibrated, use calibrated region
        if self.is_calibrated and self.calibrated_region is not None:
            return self._use_calibrated_region(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while keeping edges
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(blurred, 30, 90)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.detection_failures += 1
            return None
        
        # Find the best fretboard candidate
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            area = w_rect * h_rect
            
            # Filter by area
            area_ratio = area / frame_area
            if area_ratio < FRETBOARD_MIN_AREA_RATIO or area_ratio > FRETBOARD_MAX_AREA_RATIO:
                continue
            
            # Filter by aspect ratio (fretboard should be wider than tall)
            aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
            if aspect_ratio < FRETBOARD_ASPECT_RATIO_MIN or aspect_ratio > FRETBOARD_ASPECT_RATIO_MAX:
                continue
            
            # Score this candidate
            score = self._score_fretboard_candidate(
                (x, y, w_rect, h_rect), area_ratio, aspect_ratio, frame
            )
            
            if score > best_score:
                best_score = score
                best_candidate = {
                    'bbox': (x, y, w_rect, h_rect),
                    'area_ratio': area_ratio,
                    'aspect_ratio': aspect_ratio,
                    'score': score
                }
        
        if best_candidate:
            self.detection_failures = 0
            self.last_fretboard = best_candidate
            
            # Add to calibration history
            if self.enable_calibration and not self.is_calibrated:
                self._update_calibration(best_candidate)
            
            return best_candidate
        
        self.detection_failures += 1
        
        # If we have a recent detection, use it
        if self.last_fretboard and self.detection_failures < 5:
            return self.last_fretboard
        
        return None
    
    def _score_fretboard_candidate(self, bbox: Tuple, area_ratio: float, 
                                   aspect_ratio: float, frame: np.ndarray) -> float:
        """
        Score a fretboard candidate based on multiple factors
        
        Higher score = better candidate
        """
        score = 0.0
        
        # Prefer regions in center or slightly off-center
        x, y, w_rect, h_rect = bbox
        frame_h, frame_w = frame.shape[:2]
        center_x = x + w_rect / 2
        center_y = y + h_rect / 2
        
        # Horizontal centering (slight preference for center)
        h_center_dist = abs(center_x - frame_w / 2) / frame_w
        score += (1 - h_center_dist) * 20
        
        # Vertical position (prefer middle to upper-middle)
        v_position = center_y / frame_h
        if 0.3 < v_position < 0.7:
            score += 30
        elif 0.2 < v_position < 0.8:
            score += 20
        
        # Area preference (moderate size)
        optimal_area_ratio = 0.4
        area_score = 1 - abs(area_ratio - optimal_area_ratio) / optimal_area_ratio
        score += area_score * 25
        
        # Aspect ratio preference
        optimal_aspect = 3.5
        aspect_score = 1 - abs(aspect_ratio - optimal_aspect) / optimal_aspect
        score += aspect_score * 25
        
        return score
    
    def _update_calibration(self, fretboard_info: dict):
        """Update calibration with new detection"""
        self.calibration_history.append(fretboard_info)
        
        # Check if we have enough frames
        if len(self.calibration_history) >= CALIBRATION_FRAMES_REQUIRED:
            # Check consistency
            bboxes = [f['bbox'] for f in self.calibration_history]
            
            # Calculate average bbox
            avg_bbox = tuple(int(np.mean([b[i] for b in bboxes])) for i in range(4))
            
            # Calculate variance to check consistency
            variances = [np.std([b[i] for b in bboxes]) for i in range(4)]
            max_variance = max(variances)
            
            # If variance is low enough, we're calibrated
            threshold = 50  # pixels
            if max_variance < threshold:
                self.calibrated_region = avg_bbox
                self.is_calibrated = True
                print(f"  ✓ Fretboard calibrated: {avg_bbox}")
    
    def _use_calibrated_region(self, frame: np.ndarray) -> dict:
        """Use the calibrated region with minor adjustments"""
        x, y, w, h = self.calibrated_region
        
        # Return calibrated region
        return {
            'bbox': self.calibrated_region,
            'area_ratio': (w * h) / (frame.shape[0] * frame.shape[1]),
            'aspect_ratio': w / h if h > 0 else 0,
            'score': 100,  # High confidence for calibrated region
            'calibrated': True
        }
    
    def detect_strings(self, frame: np.ndarray, fretboard_info: dict) -> Optional[List[Tuple]]:
        """
        Detect string lines within the fretboard region
        
        Args:
            frame: RGB image frame
            fretboard_info: Fretboard detection info
            
        Returns:
            List of string lines as (x1, y1, x2, y2) or None
        """
        if not fretboard_info:
            return None
        
        x, y, w, h = fretboard_info['bbox']
        
        # Extract fretboard region
        fretboard_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(fretboard_roi, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_roi)
        
        # Apply edge detection (focus on horizontal edges for strings)
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=STRING_DETECTION_SENSITIVITY,
            minLineLength=int(w * STRING_MIN_LENGTH_RATIO),
            maxLineGap=20
        )
        
        if lines is None:
            return None
        
        # Filter for mostly horizontal lines (strings)
        string_candidates = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Keep lines that are mostly horizontal (within 15 degrees)
            if angle < 15 or angle > 165:
                # Convert back to frame coordinates
                string_candidates.append((
                    x + x1, y + y1,
                    x + x2, y + y2,
                    (y1 + y2) / 2  # Average y position
                ))
        
        # Group lines by y-position to find distinct strings
        if not string_candidates:
            return None
        
        # Sort by y position
        string_candidates.sort(key=lambda l: l[4])
        
        # Cluster nearby lines (same string)
        strings = []
        current_cluster = [string_candidates[0]]
        
        for i in range(1, len(string_candidates)):
            prev_y = current_cluster[-1][4]
            curr_y = string_candidates[i][4]
            
            # If y-positions are close, same string
            if abs(curr_y - prev_y) < h * 0.05:  # Within 5% of fretboard height
                current_cluster.append(string_candidates[i])
            else:
                # New string, save previous cluster
                # Average the lines in this cluster
                avg_line = self._average_lines(current_cluster)
                strings.append(avg_line)
                current_cluster = [string_candidates[i]]
        
        # Don't forget the last cluster
        if current_cluster:
            avg_line = self._average_lines(current_cluster)
            strings.append(avg_line)
        
        # Keep only the best NUM_STRINGS candidates
        if len(strings) > NUM_STRINGS:
            # Sort by length and keep longest
            strings.sort(key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), reverse=True)
            strings = strings[:NUM_STRINGS]
        
        # Sort by y-position (top to bottom)
        strings.sort(key=lambda l: (l[1] + l[3]) / 2)
        
        return strings
    
    def _average_lines(self, lines: List[Tuple]) -> Tuple:
        """Average multiple line segments"""
        if not lines:
            return None
        
        avg_x1 = int(np.mean([l[0] for l in lines]))
        avg_y1 = int(np.mean([l[1] for l in lines]))
        avg_x2 = int(np.mean([l[2] for l in lines]))
        avg_y2 = int(np.mean([l[3] for l in lines]))
        
        return (avg_x1, avg_y1, avg_x2, avg_y2)
    
    def draw_fretboard_region(self, frame: np.ndarray, fretboard_info: dict) -> np.ndarray:
        """Draw detected fretboard region on frame"""
        annotated_frame = frame.copy()
        
        if fretboard_info:
            x, y, w, h = fretboard_info['bbox']
            
            # Color based on calibration status
            color = (0, 255, 0) if fretboard_info.get('calibrated', False) else (255, 165, 0)
            thickness = 3 if fretboard_info.get('calibrated', False) else 2
            
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label = "Fretboard (Calibrated)" if fretboard_info.get('calibrated', False) else "Fretboard"
            label += f" [{fretboard_info['score']:.0f}]"
            
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return annotated_frame
    
    def draw_strings(self, frame: np.ndarray, strings: List[Tuple]) -> np.ndarray:
        """Draw detected string lines on frame"""
        annotated_frame = frame.copy()
        
        if strings:
            string_names = ['e', 'B', 'G', 'D', 'A', 'E']  # High to low
            
            for i, string_line in enumerate(strings):
                x1, y1, x2, y2 = string_line
                
                # Draw string line
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Add string label
                if i < len(string_names):
                    label = string_names[i]
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 - 20, y1 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )
        
        return annotated_frame
    
    def reset_calibration(self):
        """Reset calibration state"""
        self.is_calibrated = False
        self.calibration_history.clear()
        self.calibrated_region = None
        self.detection_failures = 0
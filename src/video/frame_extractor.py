"""
Extract frames from video files for analysis
"""
import cv2
import numpy as np
from pathlib import Path


class FrameExtractor:
    """Extract and preprocess video frames"""
    
    def __init__(self, target_fps=10):
        """
        Args:
            target_fps: Target frame rate for extraction (lower = faster processing)
        """
        self.target_fps = target_fps
    
    def extract_frames(self, video_path: Path, max_frames=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract (None = all)
            
        Returns:
            List of (frame, timestamp) tuples
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate frame skip interval
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        print(f"Video info: {original_fps:.2f} fps, {total_frames} frames, {duration:.2f}s")
        print(f"Extracting every {frame_interval} frames (target: {self.target_fps} fps)")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frames at target interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / original_fps
                
                # Convert BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append({
                    'frame': frame_rgb,
                    'timestamp': timestamp,
                    'frame_number': frame_count
                })
                
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames")
        
        return frames
    
    def save_frame(self, frame: np.ndarray, output_path: Path):
        """Save a single frame as image"""
        # Convert RGB back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), frame_bgr)
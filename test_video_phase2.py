#!/usr/bin/env python3
"""
Test script for Phase 2.1: Basic video processing and hand detection
"""
import sys
from pathlib import Path
import cv2

from src.video.frame_extractor import FrameExtractor
from src.video.hand_tracker import HandTracker
from src.video.fretboard_detector import FretboardDetector


def test_video_processing(video_path: str):
    """Test video processing pipeline"""
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 60)
    print("Phase 2.1: Video Processing Test")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/4] Initializing video processing components...")
    extractor = FrameExtractor(target_fps=5)  # 5 fps for testing
    hand_tracker = HandTracker()
    fretboard_detector = FretboardDetector()
    
    # Extract frames
    print("\n[2/4] Extracting frames...")
    frames = extractor.extract_frames(video_path, max_frames=10)  # Only 10 frames for test
    
    if not frames:
        print("Error: No frames extracted")
        return
    
    # Process frames
    print("\n[3/4] Processing frames...")
    output_dir = Path("data/debug")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    hand_detection_count = 0
    
    for i, frame_data in enumerate(frames):
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        # Detect hands
        hands = hand_tracker.detect_hands(frame)
        
        # Detect fretboard (placeholder)
        fretboard = fretboard_detector.detect_fretboard(frame)
        
        # Draw annotations
        annotated = hand_tracker.draw_hands_on_frame(frame, hands)
        annotated = fretboard_detector.draw_fretboard_region(annotated, fretboard)
        
        # Save debug frame
        output_path = output_dir / f"frame_{i:03d}_t{timestamp:.2f}s.jpg"
        extractor.save_frame(annotated, output_path)
        
        if hands:
            hand_detection_count += 1
            print(f"  Frame {i}: {len(hands)} hand(s) detected at {timestamp:.2f}s")
            for hand in hands:
                fingertips = hand_tracker.get_fingertip_positions(hand)
                print(f"    - {hand['label']} hand (confidence: {hand['score']:.2f})")
        else:
            print(f"  Frame {i}: No hands detected at {timestamp:.2f}s")
    
    # Results
    print("\n[4/4] Results:")
    print(f"  Total frames processed: {len(frames)}")
    print(f"  Frames with hands detected: {hand_detection_count}")
    print(f"  Detection rate: {hand_detection_count/len(frames)*100:.1f}%")
    print(f"\n  Debug frames saved to: {output_dir}")
    print("\n  Check the debug frames to see hand tracking visualization!")
    
    # Cleanup
    hand_tracker.close()
    
    print("\n" + "=" * 60)
    print("Phase 2.1 test complete!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_phase2.py <video_file>")
        print("Example: python test_video_phase2.py data/input/guitar_video.mp4")
        sys.exit(1)
    
    test_video_processing(sys.argv[1])
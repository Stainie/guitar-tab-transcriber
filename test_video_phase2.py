#!/usr/bin/env python3
"""
Test script for Phase 2.2: Fretboard detection and string detection
"""
import sys
from pathlib import Path
import cv2

from src.video.frame_extractor import FrameExtractor
from src.video.hand_tracker import HandTracker
from src.video.fretboard_detector import FretboardDetector


def test_fretboard_detection(video_path: str, max_frames=30):
    """Test fretboard and string detection"""
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 60)
    print("Phase 2.2: Fretboard Detection Test")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    extractor = FrameExtractor(target_fps=10)
    hand_tracker = HandTracker()
    fretboard_detector = FretboardDetector(enable_calibration=True)
    
    # Extract frames
    print("\n[2/5] Extracting frames...")
    frames = extractor.extract_frames(video_path, max_frames=max_frames)
    
    if not frames:
        print("Error: No frames extracted")
        return
    
    # Process frames
    print("\n[3/5] Detecting fretboard and strings...")
    output_dir = Path("data/debug/phase2_2")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fretboard_detection_count = 0
    string_detection_count = 0
    calibration_frame = None
    
    for i, frame_data in enumerate(frames):
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        # Detect fretboard
        fretboard = fretboard_detector.detect_fretboard(frame)
        
        # Detect strings if fretboard found
        strings = None
        if fretboard:
            fretboard_detection_count += 1
            strings = fretboard_detector.detect_strings(frame, fretboard)
            if strings:
                string_detection_count += 1
        
        # Detect hands (for complete visualization)
        hands = hand_tracker.detect_hands(frame)
        
        # Draw all annotations
        annotated = frame.copy()
        annotated = fretboard_detector.draw_fretboard_region(annotated, fretboard)
        if strings:
            annotated = fretboard_detector.draw_strings(annotated, strings)
        annotated = hand_tracker.draw_hands_on_frame(annotated, hands)
        
        # Save debug frame
        output_path = output_dir / f"frame_{i:03d}_t{timestamp:.2f}s.jpg"
        extractor.save_frame(annotated, output_path)
        
        # Log progress
        status = "✓ Fretboard" if fretboard else "✗ No fretboard"
        if strings:
            status += f" + {len(strings)} strings"
        if fretboard and fretboard.get('calibrated'):
            status += " [CALIBRATED]"
            if calibration_frame is None:
                calibration_frame = i
        
        print(f"  Frame {i:2d} ({timestamp:5.2f}s): {status}")
    
    # Results
    print("\n[4/5] Detection Statistics:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Fretboard detected: {fretboard_detection_count}/{len(frames)} ({fretboard_detection_count/len(frames)*100:.1f}%)")
    print(f"  Strings detected: {string_detection_count}/{len(frames)} ({string_detection_count/len(frames)*100:.1f}%)")
    
    if fretboard_detector.is_calibrated:
        print(f"  ✓ Calibration successful at frame {calibration_frame}")
        print(f"    Calibrated region: {fretboard_detector.calibrated_region}")
    else:
        print(f"  ✗ Calibration not achieved (need consistent detection)")
    
    print(f"\n[5/5] Output:")
    print(f"  Debug frames saved to: {output_dir}")
    print(f"  Review frames to verify detection quality")
    
    # Cleanup
    hand_tracker.close()
    
    print("\n" + "=" * 60)
    print("Phase 2.2 test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review debug frames in data/debug/phase2_2/")
    print("  2. Check if fretboard and strings are detected accurately")
    print("  3. If detection is poor, adjust camera angle or lighting")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_phase2.py <video_file> [max_frames]")
        print("Example: python test_video_phase2.py data/input/guitar_video.mp4 50")
        sys.exit(1)
    
    video_file = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    test_fretboard_detection(video_file, max_frames)
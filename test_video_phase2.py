#!/usr/bin/env python3
"""
Test script for Phase 2.3: Finger position tracking
"""
import sys
from pathlib import Path
import cv2

from src.video.frame_extractor import FrameExtractor
from src.video.hand_tracker import HandTracker
from src.video.fretboard_detector import FretboardDetector
from src.video.finger_mapper import FingerMapper


def test_finger_tracking(video_path: str, max_frames=50):
    """Test complete finger position tracking"""
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 60)
    print("Phase 2.3: Finger Position Tracking Test")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/6] Initializing components...")
    extractor = FrameExtractor(target_fps=10)
    hand_tracker = HandTracker()
    fretboard_detector = FretboardDetector(enable_calibration=True)
    finger_mapper = FingerMapper()
    
    # Extract frames
    print("\n[2/6] Extracting frames...")
    frames = extractor.extract_frames(video_path, max_frames=max_frames)
    
    if not frames:
        print("Error: No frames extracted")
        return
    
    # Process frames
    print("\n[3/6] Processing frames...")
    output_dir = Path("data/debug/phase2_3")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    calibration_done = False
    notes_detected_count = 0
    finger_detection_count = 0
    
    all_detected_notes = []
    
    for i, frame_data in enumerate(frames):
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_shape = frame.shape
        
        # Detect fretboard
        fretboard = fretboard_detector.detect_fretboard(frame)
        
        # Detect strings
        strings = None
        if fretboard:
            strings = fretboard_detector.detect_strings(frame, fretboard)
        
        # Calibrate finger mapper
        if not calibration_done and fretboard and strings:
            if finger_mapper.calibrate(fretboard, strings, frame_shape):
                calibration_done = True
                print(f"  ✓ Finger mapper calibrated at frame {i}")
        
        # Detect hands
        hands = hand_tracker.detect_hands(frame)
        
        # Map hands to fretboard
        hand_mappings = []
        if finger_mapper.is_calibrated():
            for hand in hands:
                mapping = finger_mapper.map_hand_to_fretboard(hand, frame_shape)
                if mapping:
                    hand_mappings.append(mapping)
                    finger_detection_count += 1
        
        # Get played notes
        played_notes = finger_mapper.get_played_notes(hand_mappings)
        
        # Draw all annotations
        annotated = frame.copy()
        annotated = fretboard_detector.draw_fretboard_region(annotated, fretboard)
        if strings:
            annotated = fretboard_detector.draw_strings(annotated, strings)
        annotated = finger_mapper.draw_fret_markers(annotated)
        annotated = hand_tracker.draw_hands_on_frame(annotated, hands)
        annotated = finger_mapper.draw_finger_mappings(annotated, hand_mappings)
        
        # Add played notes info to frame
        if played_notes:
            notes_detected_count += 1
            y_offset = 30
            cv2.putText(
                annotated,
                f"Playing: {len(played_notes)} note(s)",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            for note in played_notes:
                y_offset += 25
                note_str = f"  String {note['string']+1}, Fret {note['fret']} ({note['finger']})"
                cv2.putText(
                    annotated,
                    note_str,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
            # Store for analysis
            all_detected_notes.append({
                'timestamp': timestamp,
                'notes': played_notes
            })
        
        # Save debug frame
        output_path = output_dir / f"frame_{i:03d}_t{timestamp:.2f}s.jpg"
        extractor.save_frame(annotated, output_path)
        
        # Log progress
        status = []
        if fretboard:
            status.append("✓ FB")
        if strings:
            status.append(f"{len(strings)}str")
        if hand_mappings:
            status.append(f"{len(hand_mappings)}hand")
        if played_notes:
            status.append(f"♪{len(played_notes)}notes")
        
        status_str = " | ".join(status) if status else "No detection"
        print(f"  Frame {i:2d} ({timestamp:5.2f}s): {status_str}")
    
    # Analysis
    print("\n[4/6] Detection Statistics:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Finger mapper calibrated: {'Yes' if calibration_done else 'No'}")
    print(f"  Frames with finger detections: {finger_detection_count}")
    print(f"  Frames with notes detected: {notes_detected_count}")
    
    if notes_detected_count > 0:
        print(f"  Detection rate: {notes_detected_count/len(frames)*100:.1f}%")
    
    # Note statistics
    if all_detected_notes:
        print("\n[5/6] Note Analysis:")
        
        # Count unique string/fret combinations
        unique_notes = set()
        for detection in all_detected_notes:
            for note in detection['notes']:
                unique_notes.add((note['string'], note['fret']))
        
        print(f"  Unique notes detected: {len(unique_notes)}")
        print(f"  Most common positions:")
        
        # Count frequencies
        from collections import Counter
        note_counts = Counter()
        for detection in all_detected_notes:
            for note in detection['notes']:
                note_counts[(note['string'], note['fret'])] += 1
        
        for (string, fret), count in note_counts.most_common(5):
            print(f"    String {string+1}, Fret {fret}: {count} times")
    else:
        print("\n[5/6] Note Analysis:")
        print("  No notes detected")
        print("  Tips for better detection:")
        print("    - Ensure hands are visible and well-lit")
        print("    - Keep fretboard in frame")
        print("    - Use a clear camera angle showing fingers on frets")
    
    print(f"\n[6/6] Output:")
    print(f"  Debug frames saved to: {output_dir}")
    print(f"  Review frames to verify finger tracking")
    
    # Cleanup
    hand_tracker.close()
    
    print("\n" + "=" * 60)
    print("Phase 2.3 test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review debug frames to check finger-to-fret mapping accuracy")
    print("  2. Test with different camera angles if detection is poor")
    print("  3. Proceed to Phase 2.4 for string state detection")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_phase2.py <video_file> [max_frames]")
        print("Example: python test_video_phase2.py data/input/guitar_video.mp4 50")
        sys.exit(1)
    
    video_file = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    test_finger_tracking(video_file, max_frames)
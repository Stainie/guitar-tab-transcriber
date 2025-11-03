#!/usr/bin/env python3
"""
Test script for Complete Phase 2: Full video analysis pipeline
"""
import sys
from pathlib import Path
import cv2

from src.video.frame_extractor import FrameExtractor
from src.video.hand_tracker import HandTracker
from src.video.fretboard_detector import FretboardDetector
from src.video.finger_mapper import FingerMapper
from src.video.string_state_detector import StringStateDetector


def test_complete_phase2(video_path: str, max_frames=100):
    """Test complete Phase 2 pipeline with all components"""
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 70)
    print("PHASE 2 COMPLETE: Full Video Analysis Pipeline Test")
    print("=" * 70)
    
    # Initialize all components
    print("\n[1/7] Initializing all Phase 2 components...")
    extractor = FrameExtractor(target_fps=15)  # Higher fps for better motion detection
    hand_tracker = HandTracker()
    fretboard_detector = FretboardDetector(enable_calibration=True)
    finger_mapper = FingerMapper()
    string_detector = StringStateDetector(history_length=5)
    
    # Extract frames
    print("\n[2/7] Extracting video frames...")
    frames = extractor.extract_frames(video_path, max_frames=max_frames)
    
    if not frames:
        print("Error: No frames extracted")
        return
    
    # Setup output
    output_dir = Path("data/debug/phase2_complete")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Statistics
    stats = {
        'total_frames': len(frames),
        'fretboard_detected': 0,
        'strings_detected': 0,
        'hands_detected': 0,
        'fingers_mapped': 0,
        'picking_detected': 0,
        'notes_played': 0
    }
    
    all_note_events = []
    calibration_done = False
    
    # Process all frames
    print("\n[3/7] Processing frames with full pipeline...")
    
    for i, frame_data in enumerate(frames):
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_shape = frame.shape
        
        # Step 1: Detect fretboard
        fretboard = fretboard_detector.detect_fretboard(frame)
        if fretboard:
            stats['fretboard_detected'] += 1
        
        # Step 2: Detect strings
        strings = None
        if fretboard:
            strings = fretboard_detector.detect_strings(frame, fretboard)
            if strings:
                stats['strings_detected'] += 1
        
        # Step 3: Calibrate finger mapper
        if not calibration_done and fretboard and strings:
            if finger_mapper.calibrate(fretboard, strings, frame_shape):
                calibration_done = True
                print(f"  ✓ System calibrated at frame {i} ({timestamp:.2f}s)")
        
        # Step 4: Detect hands
        hands = hand_tracker.detect_hands(frame)
        if hands:
            stats['hands_detected'] += 1
        
        # Step 5: Map fingers to fretboard
        hand_mappings = []
        if finger_mapper.is_calibrated():
            for hand in hands:
                mapping = finger_mapper.map_hand_to_fretboard(hand, frame_shape)
                if mapping:
                    hand_mappings.append(mapping)
                    stats['fingers_mapped'] += 1
        
        # Step 6: Get fretted notes
        played_notes = finger_mapper.get_played_notes(hand_mappings)
        
        # Step 7: Detect string activity (picking/strumming)
        string_activity = string_detector.detect_string_activity(
            hands, hand_mappings, strings, frame_shape, timestamp
        )
        
        if string_activity['picking_detected']:
            stats['picking_detected'] += 1
        
        # Step 8: Combine fretting + picking = complete note events
        note_events = string_detector.combine_with_fretting(string_activity, played_notes)
        
        if note_events:
            stats['notes_played'] += 1
            all_note_events.append({
                'timestamp': timestamp,
                'notes': note_events,
                'strumming': string_activity['strumming']
            })
        
        # Visualize everything
        annotated = frame.copy()
        annotated = fretboard_detector.draw_fretboard_region(annotated, fretboard)
        if strings:
            annotated = fretboard_detector.draw_strings(annotated, strings)
        annotated = finger_mapper.draw_fret_markers(annotated)
        annotated = hand_tracker.draw_hands_on_frame(annotated, hands)
        annotated = finger_mapper.draw_finger_mappings(annotated, hand_mappings)
        annotated = string_detector.draw_string_activity(annotated, string_activity, strings)
        
        # Add comprehensive status overlay
        y_offset = 30
        
        # Calibration status
        status_color = (0, 255, 0) if calibration_done else (0, 165, 255)
        cv2.putText(annotated, f"Calibrated: {calibration_done}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Note events
        if note_events:
            y_offset += 30
            mode = "Strumming" if string_activity['strumming'] else "Playing"
            cv2.putText(annotated, f"{mode}: {len(note_events)} note(s)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for note in note_events[:5]:  # Show max 5 notes
                y_offset += 25
                note_str = f"  S{note['string']+1}F{note['fret']}"
                if note.get('finger'):
                    note_str += f" ({note['finger'][0].upper()})"
                cv2.putText(annotated, note_str, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save frame
        output_path = output_dir / f"frame_{i:04d}_t{timestamp:.2f}s.jpg"
        extractor.save_frame(annotated, output_path)
        
        # Progress indicator every 10 frames
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames...")
    
    # Analysis and reporting
    print("\n[4/7] Pipeline Statistics:")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Fretboard detected: {stats['fretboard_detected']} ({stats['fretboard_detected']/stats['total_frames']*100:.1f}%)")
    print(f"  Strings detected: {stats['strings_detected']} ({stats['strings_detected']/stats['total_frames']*100:.1f}%)")
    print(f"  Hands detected: {stats['hands_detected']} ({stats['hands_detected']/stats['total_frames']*100:.1f}%)")
    print(f"  Finger mappings: {stats['fingers_mapped']}")
    print(f"  Picking detected: {stats['picking_detected']} ({stats['picking_detected']/stats['total_frames']*100:.1f}%)")
    print(f"  Notes played: {stats['notes_played']} frames")
    
    print("\n[5/7] Note Event Analysis:")
    if all_note_events:
        total_notes = sum(len(event['notes']) for event in all_note_events)
        strum_count = sum(1 for event in all_note_events if event['strumming'])
        pick_count = len(all_note_events) - strum_count
        
        print(f"  Total note events: {len(all_note_events)}")
        print(f"  Total individual notes: {total_notes}")
        print(f"  Strumming events: {strum_count}")
        print(f"  Picking events: {pick_count}")
        
        # Analyze note frequency
        from collections import Counter
        note_counts = Counter()
        for event in all_note_events:
            for note in event['notes']:
                note_counts[(note['string'], note['fret'])] += 1
        
        print(f"\n  Most played positions:")
        for (string, fret), count in note_counts.most_common(10):
            string_names = ['e', 'B', 'G', 'D', 'A', 'E']
            string_name = string_names[string] if string < len(string_names) else f"S{string+1}"
            print(f"    {string_name} string, Fret {fret}: {count} times")
    else:
        print("  No note events detected")
        print("\n  Troubleshooting tips:")
        print("    - Ensure both hands are visible")
        print("    - Check that picking motions are clear")
        print("    - Verify fretboard is well-lit and in frame")
        print("    - Try a clearer camera angle")
    
    print(f"\n[6/7] Output Files:")
    print(f"  Debug frames: {output_dir}")
    print(f"  Total files: {len(list(output_dir.glob('*.jpg')))}")
    
    # Save summary report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 2 Complete: Video Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames analyzed: {stats['total_frames']}\n")
        f.write(f"Duration: {frames[-1]['timestamp']:.2f}s\n\n")
        
        f.write("Detection Statistics:\n")
        for key, value in stats.items():
            if key != 'total_frames':
                pct = value / stats['total_frames'] * 100
                f.write(f"  {key}: {value} ({pct:.1f}%)\n")
        
        f.write("\nNote Events:\n")
        for event in all_note_events[:20]:  # First 20 events
            mode = "STRUM" if event['strumming'] else "PICK"
            f.write(f"  [{event['timestamp']:.2f}s] {mode}: ")
            notes_str = ", ".join([f"S{n['string']+1}F{n['fret']}" for n in event['notes']])
            f.write(f"{notes_str}\n")
    
    print(f"  Analysis report: {report_path}")
    
    print("\n[7/7] Phase 2 Complete!")
    
    # Cleanup
    hand_tracker.close()
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE - All video analysis components working!")
    print("=" * 70)
    print("\nNext Phase: Phase 3 - Multimodal Fusion")
    print("  Combine video analysis with audio analysis for maximum accuracy")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_phase2.py <video_file> [max_frames]")
        print("Example: python test_video_phase2.py data/input/guitar_video.mp4 100")
        sys.exit(1)
    
    video_file = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    test_complete_phase2(video_file, max_frames)
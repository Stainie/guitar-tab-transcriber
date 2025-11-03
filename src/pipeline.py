"""
Complete transcription pipeline - Phase 3 (Multimodal Fusion)
"""
from pathlib import Path
import numpy as np

# Phase 1: Audio
from src.audio.extractor import AudioExtractor
from src.audio.pitch_detector import PitchDetector
from src.audio.onset_detector import OnsetDetector

# Phase 2: Video
from src.video.frame_extractor import FrameExtractor
from src.video.hand_tracker import HandTracker
from src.video.fretboard_detector import FretboardDetector
from src.video.finger_mapper import FingerMapper
from src.video.string_state_detector import StringStateDetector

# Phase 3: Fusion
from src.fusion.multimodal_fusion import MultimodalFusion
from src.fusion.position_optimizer import PositionOptimizer

# Tab generation
from src.transcription.guitar_mapper import GuitarMapper
from src.transcription.tab_generator import TabGenerator


class TranscriptionPipeline:
    """
    Complete transcription pipeline with multimodal fusion
    """
    
    def __init__(self, 
                 mode='multimodal',  # 'audio', 'video', or 'multimodal'
                 audio_weight=0.4,
                 video_weight=0.6):
        """
        Args:
            mode: Processing mode
            audio_weight: Weight for audio in fusion
            video_weight: Weight for video in fusion
        """
        self.mode = mode
        
        # Phase 1: Audio components
        if mode in ['audio', 'multimodal']:
            self.audio_extractor = AudioExtractor()
            self.pitch_detector = PitchDetector()
            self.onset_detector = OnsetDetector()
            self.guitar_mapper = GuitarMapper()
        
        # Phase 2: Video components
        if mode in ['video', 'multimodal']:
            self.frame_extractor = FrameExtractor(target_fps=15)
            self.hand_tracker = HandTracker()
            self.fretboard_detector = FretboardDetector(enable_calibration=True)
            self.finger_mapper = FingerMapper()
            self.string_detector = StringStateDetector()
        
        # Phase 3: Fusion components
        if mode == 'multimodal':
            self.fusion = MultimodalFusion(audio_weight, video_weight)
            self.position_optimizer = PositionOptimizer()
        
        # Tab generation
        self.tab_generator = TabGenerator()
    
    def process(self, input_path: str, output_debug: bool = False) -> dict:
        """
        Process an audio/video file and return guitar tab with metadata
        
        Args:
            input_path: Path to input file
            output_debug: Whether to save debug visualizations
            
        Returns:
            Dictionary with tab, notes, and metadata
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\n{'='*70}")
        print(f"PHASE 3: MULTIMODAL FUSION TRANSCRIPTION")
        print(f"Mode: {self.mode.upper()}")
        print(f"{'='*70}\n")
        
        result = {
            'audio_notes': None,
            'video_notes': None,
            'fused_notes': None,
            'tab': None,
            'metadata': {}
        }
        
        # Phase 1: Audio Analysis
        if self.mode in ['audio', 'multimodal']:
            print("=" * 70)
            print("PHASE 1: AUDIO ANALYSIS")
            print("=" * 70)
            audio_notes = self._process_audio(input_path)
            result['audio_notes'] = audio_notes
            print(f"✓ Audio analysis complete: {len(audio_notes)} notes detected\n")
        
        # Phase 2: Video Analysis
        if self.mode in ['video', 'multimodal']:
            print("=" * 70)
            print("PHASE 2: VIDEO ANALYSIS")
            print("=" * 70)
            video_notes, video_context = self._process_video(input_path, output_debug)
            result['video_notes'] = video_notes
            result['metadata']['video_context'] = video_context
            print(f"✓ Video analysis complete: {len(video_notes)} note events detected\n")
        
        # Phase 3: Fusion
        if self.mode == 'multimodal':
            print("=" * 70)
            print("PHASE 3: MULTIMODAL FUSION")
            print("=" * 70)
            
            fused_notes = self.fusion.fuse_predictions(
                result['audio_notes'],
                result['video_notes'],
                audio_context={'onset_strength': 5.0},  # Could be calculated
                video_context=result['metadata'].get('video_context', {})
            )
            
            print(f"Step 1: Fusion complete - {len(fused_notes)} notes after fusion")
            
            # Optimize positions
            optimized_notes = self.position_optimizer.optimize_positions(fused_notes)
            print(f"Step 2: Position optimization complete")
            
            # Group into chords
            note_events = self.position_optimizer.group_into_chords(optimized_notes)
            print(f"Step 3: Chord grouping complete - {len(note_events)} events")
            
            # Get fusion statistics
            fusion_stats = self.fusion.get_fusion_stats(fused_notes)
            result['metadata']['fusion_stats'] = fusion_stats
            
            print(f"\nFusion Statistics:")
            print(f"  Total notes: {fusion_stats.get('total_notes', 0)}")
            print(f"  Average confidence: {fusion_stats.get('average_confidence', 0):.2f}")
            print(f"  Corrections made: {fusion_stats.get('corrections_made', 0)}")
            print(f"  Conflicts resolved: {fusion_stats.get('conflicts_resolved', 0)}")
            print(f"\nSources breakdown:")
            for source, count in fusion_stats.get('sources', {}).items():
                print(f"    {source}: {count}")
            
            result['fused_notes'] = optimized_notes
            result['note_events'] = note_events
            
            # Generate tab from fused notes
            notes_for_tab = optimized_notes
        elif self.mode == 'audio':
            notes_for_tab = result['audio_notes']
        else:  # video only
            notes_for_tab = result['video_notes']
        
        # Generate final tab
        print(f"\n{'='*70}")
        print("GENERATING TAB")
        print(f"{'='*70}")
        
        tab = self.tab_generator.generate(notes_for_tab)
        result['tab'] = tab
        
        print("✓ Tab generation complete\n")
        
        return result
    
    def _process_audio(self, input_path: Path) -> list:
        """Process audio (Phase 1)"""
        print("Step 1/5: Extracting audio...")
        audio, sample_rate = self.audio_extractor.extract(input_path)
        print(f"  → Audio loaded: {len(audio)/sample_rate:.2f}s @ {sample_rate} Hz")
        
        print("Step 2/5: Detecting pitches...")
        pitches, confidences, times = self.pitch_detector.detect(audio, sample_rate)
        print(f"  → Detected {len(pitches)} pitch frames")
        
        print("Step 3/5: Detecting note onsets...")
        onset_times = self.onset_detector.detect(audio, sample_rate)
        print(f"  → Found {len(onset_times)} note onsets")
        
        print("Step 4/5: Mapping to guitar strings and frets...")
        notes = self.guitar_mapper.map_to_guitar(
            pitches, confidences, times, onset_times
        )
        print(f"  → Mapped {len(notes)} notes")
        
        print("Step 5/5: Audio analysis complete")
        
        return notes
    
    def _process_video(self, input_path: Path, output_debug: bool) -> tuple:
        """Process video (Phase 2)"""
        print("Step 1/7: Extracting frames...")
        frames = self.frame_extractor.extract_frames(input_path, max_frames=200)
        print(f"  → Extracted {len(frames)} frames")
        
        video_notes = []
        calibration_done = False
        debug_frames_saved = 0
        
        video_context = {
            'calibration_quality': 0.0,
            'total_frames': len(frames),
            'frames_with_detection': 0
        }
        
        print("Step 2/7: Analyzing frames...")
        
        for i, frame_data in enumerate(frames):
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            frame_shape = frame.shape
            
            # Detect fretboard
            fretboard = self.fretboard_detector.detect_fretboard(frame)
            
            # Detect strings
            strings = None
            if fretboard:
                strings = self.fretboard_detector.detect_strings(frame, fretboard)
            
            # Calibrate
            if not calibration_done and fretboard and strings:
                if self.finger_mapper.calibrate(fretboard, strings, frame_shape):
                    calibration_done = True
                    video_context['calibration_frame'] = i
                    video_context['calibration_quality'] = 0.9
                    print(f"  → Calibrated at frame {i}")
            
            # Detect hands
            hands = self.hand_tracker.detect_hands(frame)
            
            # Map fingers
            hand_mappings = []
            if self.finger_mapper.is_calibrated():
                for hand in hands:
                    mapping = self.finger_mapper.map_hand_to_fretboard(hand, frame_shape)
                    if mapping:
                        hand_mappings.append(mapping)
            
            # Get fretted notes
            played_notes = self.finger_mapper.get_played_notes(hand_mappings)
            
            # Detect picking
            string_activity = self.string_detector.detect_string_activity(
                hands, hand_mappings, strings, frame_shape, timestamp
            )
            
            # Combine
            note_events = self.string_detector.combine_with_fretting(
                string_activity, played_notes
            )
            
            if note_events:
                video_context['frames_with_detection'] += 1
                for note in note_events:
                    note['time'] = timestamp
                    note['timestamp'] = timestamp
                    video_notes.append(note)
            
            # Save debug frames (first 10 with detections)
            if output_debug and note_events and debug_frames_saved < 10:
                debug_dir = Path("data/debug/phase3_fusion")
                debug_dir.mkdir(exist_ok=True, parents=True)
                
                annotated = frame.copy()
                annotated = self.fretboard_detector.draw_fretboard_region(annotated, fretboard)
                if strings:
                    annotated = self.fretboard_detector.draw_strings(annotated, strings)
                annotated = self.finger_mapper.draw_fret_markers(annotated)
                annotated = self.hand_tracker.draw_hands_on_frame(annotated, hands)
                annotated = self.finger_mapper.draw_finger_mappings(annotated, hand_mappings)
                annotated = self.string_detector.draw_string_activity(annotated, string_activity, strings)
                
                output_path = debug_dir / f"detection_{debug_frames_saved:02d}_t{timestamp:.2f}s.jpg"
                self.frame_extractor.save_frame(annotated, output_path)
                debug_frames_saved += 1
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"  → Processed {i+1}/{len(frames)} frames...")
        
        print(f"Step 3/7: Frame analysis complete")
        print(f"Step 4/7: Calibration: {'✓ Success' if calibration_done else '✗ Failed'}")
        print(f"Step 5/7: Detection rate: {video_context['frames_with_detection']}/{len(frames)} frames")
        print(f"Step 6/7: Total note events: {len(video_notes)}")
        print(f"Step 7/7: Video analysis complete")
        
        if output_debug and debug_frames_saved > 0:
            print(f"  → Debug frames saved to data/debug/phase3_fusion/")
        
        # Cleanup
        self.hand_tracker.close()
        
        return video_notes, video_context
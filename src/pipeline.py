"""
Main transcription pipeline for Phase 1
"""
from pathlib import Path
import numpy as np

from src.audio.extractor import AudioExtractor
from src.audio.pitch_detector import PitchDetector
from src.audio.onset_detector import OnsetDetector
from src.transcription.guitar_mapper import GuitarMapper
from src.transcription.tab_generator import TabGenerator


class TranscriptionPipeline:
    """
    Main pipeline that orchestrates the transcription process
    """
    
    def __init__(self):
        self.audio_extractor = AudioExtractor()
        self.pitch_detector = PitchDetector()
        self.onset_detector = OnsetDetector()
        self.guitar_mapper = GuitarMapper()
        self.tab_generator = TabGenerator()
    
    def process(self, input_path: str) -> str:
        """
        Process an audio/video file and return guitar tab
        
        Args:
            input_path: Path to input audio or video file
            
        Returns:
            String containing the generated guitar tab
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"Step 1/5: Extracting audio...")
        audio, sample_rate = self.audio_extractor.extract(input_path)
        print(f"  → Audio loaded: {len(audio)/sample_rate:.2f} seconds @ {sample_rate} Hz")
        
        print(f"Step 2/5: Detecting pitches...")
        pitches, confidences, times = self.pitch_detector.detect(audio, sample_rate)
        print(f"  → Detected {len(pitches)} pitch frames")
        
        print(f"Step 3/5: Detecting note onsets...")
        onset_times = self.onset_detector.detect(audio, sample_rate)
        print(f"  → Found {len(onset_times)} note onsets")
        
        print(f"Step 4/5: Mapping to guitar strings and frets...")
        notes = self.guitar_mapper.map_to_guitar(
            pitches, confidences, times, onset_times
        )
        print(f"  → Mapped {len(notes)} notes")
        
        print(f"Step 5/5: Generating tab...")
        tab = self.tab_generator.generate(notes)
        print(f"  → Tab generated successfully")
        
        return tab
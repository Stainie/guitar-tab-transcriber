"""
Audio extraction from video/audio files
"""
import numpy as np
import librosa
from pathlib import Path


class AudioExtractor:
    """Extract and preprocess audio from various file formats"""
    
    def __init__(self, target_sr=22050):
        """
        Args:
            target_sr: Target sample rate for audio processing
        """
        self.target_sr = target_sr
    
    def extract(self, file_path: Path) -> tuple[np.ndarray, int]:
        """
        Extract audio from file (works with both audio and video files)
        
        Args:
            file_path: Path to audio or video file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # librosa can handle most audio formats
            # For video files, it uses ffmpeg under the hood
            audio, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from {file_path}: {e}")
"""
Audio extraction from video/audio files
"""
import numpy as np
import librosa
from pathlib import Path
from pydub import AudioSegment
import tempfile
import os


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
        file_path = Path(file_path)
        
        try:
            # First try librosa directly for audio files
            audio, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            # If librosa fails (e.g., for video files), use pydub with ffmpeg
            try:
                print(f"Librosa failed, trying pydub with ffmpeg for {file_path}...")
                
                # Load the video/audio file with pydub
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # For video files
                    audio_segment = AudioSegment.from_file(str(file_path), format=file_path.suffix[1:])
                else:
                    # For audio files
                    audio_segment = AudioSegment.from_file(str(file_path))
                
                # Convert to mono
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Export to temporary WAV file that librosa can read
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_path = temp_wav.name
                    audio_segment.export(temp_path, format='wav')
                
                try:
                    # Load with librosa
                    audio, sr = librosa.load(temp_path, sr=self.target_sr, mono=True)
                    
                    # Normalize audio
                    audio = librosa.util.normalize(audio)
                    
                    return audio, sr
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e2:
                raise RuntimeError(f"Failed to extract audio from {file_path}: {e2}")
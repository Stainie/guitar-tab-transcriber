"""
Onset detection using librosa
"""
import numpy as np
import librosa


class OnsetDetector:
    """Detect note onsets (attack times) in audio"""
    
    def __init__(self):
        pass
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detect onset times in audio
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate of audio
            
        Returns:
            Array of onset times in seconds
        """
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            units='frames'
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
        
        return onset_times
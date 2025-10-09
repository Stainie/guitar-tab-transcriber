"""
Pitch detection using CREPE
"""
import numpy as np
import crepe


class PitchDetector:
    """Detect pitches in audio using CREPE model"""
    
    def __init__(self, model_capacity='tiny', step_size=10):
        """
        Args:
            model_capacity: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
            step_size: Time step in milliseconds between predictions
        """
        self.model_capacity = model_capacity
        self.step_size = step_size
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect pitches in audio signal
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (frequencies, confidences, times)
        """
        # Use CREPE for pitch detection
        times, frequencies, confidences, _ = crepe.predict(
            audio,
            sample_rate,
            model_capacity=self.model_capacity,
            step_size=self.step_size,
            viterbi=True  # Use Viterbi decoding for smoother pitch tracking
        )
        
        # Filter out low-confidence predictions
        confidence_threshold = 0.5
        mask = confidences > confidence_threshold
        
        return frequencies[mask], confidences[mask], times[mask]
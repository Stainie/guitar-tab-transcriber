"""
Confidence scoring for audio and video predictions - Phase 3.1
"""
import numpy as np
from typing import Dict, List, Optional


class ConfidenceScorer:
    """Score confidence of audio and video predictions"""
    
    def __init__(self):
        # Confidence thresholds
        self.audio_min_confidence = 0.5
        self.video_min_confidence = 0.6
        
        # Weights for different factors
        self.weights = {
            'audio': {
                'pitch_confidence': 0.4,
                'onset_clarity': 0.3,
                'frequency_stability': 0.3
            },
            'video': {
                'hand_detection': 0.3,
                'finger_mapping': 0.3,
                'picking_detection': 0.2,
                'calibration_quality': 0.2
            }
        }
    
    def score_audio_prediction(self, audio_note: Dict, context: Dict) -> float:
        """
        Score confidence of an audio-detected note
        
        Args:
            audio_note: Note from audio analysis
            context: Additional context (adjacent notes, etc.)
            
        Returns:
            Confidence score (0-1)
        """
        scores = []
        
        # Pitch detection confidence (from CREPE)
        if 'confidence' in audio_note:
            pitch_conf = audio_note['confidence']
            scores.append(pitch_conf * self.weights['audio']['pitch_confidence'])
        
        # Onset clarity (how distinct the note start is)
        if 'onset_strength' in context:
            onset_clarity = min(1.0, context['onset_strength'] / 10.0)
            scores.append(onset_clarity * self.weights['audio']['onset_clarity'])
        
        # Frequency stability (less variation = more confident)
        if 'frequency_variance' in context:
            variance = context['frequency_variance']
            stability = 1.0 / (1.0 + variance / 10.0)  # Lower variance = higher score
            scores.append(stability * self.weights['audio']['frequency_stability'])
        
        # Default weights if context missing
        if not scores:
            return audio_note.get('confidence', 0.5)
        
        return sum(scores)
    
    def score_video_prediction(self, video_note: Dict, context: Dict) -> float:
        """
        Score confidence of a video-detected note
        
        Args:
            video_note: Note from video analysis
            context: Additional context (calibration state, etc.)
            
        Returns:
            Confidence score (0-1)
        """
        scores = []
        
        # Hand detection confidence
        if 'confidence' in video_note:
            hand_conf = video_note['confidence']
            scores.append(hand_conf * self.weights['video']['hand_detection'])
        
        # Finger mapping quality (is position unambiguous?)
        if 'position_clarity' in context:
            mapping_quality = context['position_clarity']
            scores.append(mapping_quality * self.weights['video']['finger_mapping'])
        
        # Picking detection (did we see the string being played?)
        if 'picking_detected' in context:
            picking_conf = 1.0 if context['picking_detected'] else 0.3
            scores.append(picking_conf * self.weights['video']['picking_detection'])
        
        # Calibration quality
        if 'calibration_quality' in context:
            calib_quality = context['calibration_quality']
            scores.append(calib_quality * self.weights['video']['calibration_quality'])
        
        # Default if context missing
        if not scores:
            return video_note.get('confidence', 0.6)
        
        return sum(scores)
    
    def compare_predictions(self, audio_note: Dict, video_note: Dict) -> Dict:
        """
        Compare audio and video predictions for the same time window
        
        Returns:
            Dictionary with comparison results
        """
        # Check if notes match
        string_match = audio_note.get('string') == video_note.get('string')
        fret_match = audio_note.get('fret') == video_note.get('fret')
        
        # Calculate frequency from video position (for comparison)
        video_freq = self._calculate_frequency_from_position(
            video_note.get('string', 0),
            video_note.get('fret', 0)
        )
        
        audio_freq = audio_note.get('frequency', 0)
        
        # Frequency difference in cents
        if audio_freq > 0 and video_freq > 0:
            freq_diff_cents = abs(1200 * np.log2(audio_freq / video_freq))
        else:
            freq_diff_cents = 999  # No comparison possible
        
        # Agreement score
        agreement = 0.0
        if string_match and fret_match:
            agreement = 1.0
        elif string_match:
            agreement = 0.5
        elif freq_diff_cents < 100:  # Within 1 semitone
            agreement = 0.7
        
        return {
            'string_match': string_match,
            'fret_match': fret_match,
            'frequency_diff_cents': freq_diff_cents,
            'agreement': agreement,
            'audio_freq': audio_freq,
            'video_freq': video_freq
        }
    
    def _calculate_frequency_from_position(self, string: int, fret: int) -> float:
        """Calculate expected frequency for a string/fret position"""
        from config.guitar_config import STANDARD_TUNING
        
        if string >= len(STANDARD_TUNING):
            return 0
        
        note, octave = STANDARD_TUNING[string]
        
        # Note to frequency conversion
        note_offsets = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6,
            'E': -5, 'F': -4, 'F#': -3, 'G': -2,
            'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        # A4 = 440 Hz
        a4_freq = 440.0
        semitones_from_a4 = note_offsets[note] + (octave - 4) * 12 + fret
        
        return a4_freq * (2 ** (semitones_from_a4 / 12))
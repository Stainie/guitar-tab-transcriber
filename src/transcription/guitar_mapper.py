"""
Map detected pitches to guitar strings and frets
"""
import numpy as np
from config.guitar_config import STANDARD_TUNING, NUM_FRETS, PITCH_TOLERANCE, FRET_PENALTY_WEIGHT


class GuitarMapper:
    """Map pitches to guitar strings and fret positions"""
    
    def __init__(self, tuning=None):
        """
        Args:
            tuning: Guitar tuning (defaults to standard tuning)
        """
        self.tuning = tuning or STANDARD_TUNING
        self._build_fretboard()
        self.last_position = None  # Track last note for position continuity
    
    def _build_fretboard(self):
        """Build a mapping of all possible notes on the fretboard"""
        self.fretboard = []
        
        for string_idx, (note, octave) in enumerate(self.tuning):
            string_notes = []
            base_freq = self._note_to_freq(note, octave)
            
            for fret in range(NUM_FRETS + 1):
                # Each fret is one semitone up
                freq = base_freq * (2 ** (fret / 12))
                string_notes.append({
                    'string': string_idx,
                    'fret': fret,
                    'frequency': freq
                })
            
            self.fretboard.append(string_notes)
    
    def _note_to_freq(self, note: str, octave: int) -> float:
        """Convert note name and octave to frequency"""
        note_offsets = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6,
            'E': -5, 'F': -4, 'F#': -3, 'G': -2,
            'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        # A4 = 440 Hz
        a4_freq = 440.0
        semitones_from_a4 = note_offsets[note] + (octave - 4) * 12
        
        return a4_freq * (2 ** (semitones_from_a4 / 12))
    
    def _calculate_position_score(self, note_info, frequency):
        """
        Score a position based on multiple factors
        Lower score is better
        """
        if note_info['frequency'] <= 0:
            return float('inf')
        
        # Frequency accuracy (in cents)
        cents_diff = abs(1200 * np.log2(frequency / note_info['frequency']))
        
        if cents_diff > PITCH_TOLERANCE:
            return float('inf')
        
        # Start with frequency error
        score = cents_diff
        
        # Penalty for high frets (prefer lower frets)
        fret_penalty = note_info['fret'] * FRET_PENALTY_WEIGHT
        score += fret_penalty
        
        # Prefer higher strings for higher notes (more natural)
        # Lower string index = lower pitch string
        # So for higher frequencies, prefer higher string indices
        if frequency > 300:  # Roughly above D4
            string_preference = (5 - note_info['string']) * 2  # Prefer strings 4, 5
        else:
            string_preference = note_info['string'] * 2  # Prefer strings 0, 1, 2
        
        score += string_preference
        
        # Continuity: prefer positions close to last note
        if self.last_position:
            string_distance = abs(note_info['string'] - self.last_position['string'])
            fret_distance = abs(note_info['fret'] - self.last_position['fret'])
            position_change_penalty = string_distance * 3 + fret_distance * 0.5
            score += position_change_penalty
        
        return score
    
    def _find_best_position(self, frequency: float):
        """Find the best string/fret combination for a frequency"""
        best_match = None
        best_score = float('inf')
        
        # Check for octave errors (common in pitch detection)
        # Try the detected frequency and its octave variants
        freq_candidates = [
            frequency,
            frequency / 2,  # One octave down
            frequency * 2,  # One octave up
        ]
        
        for test_freq in freq_candidates:
            # Skip frequencies outside guitar range
            if test_freq < 80 or test_freq > 1200:
                continue
            
            for string_notes in self.fretboard:
                for note_info in string_notes:
                    score = self._calculate_position_score(note_info, test_freq)
                    
                    if score < best_score:
                        best_score = score
                        best_match = note_info.copy()
                        best_match['detected_freq'] = frequency
                        best_match['corrected_freq'] = test_freq
        
        # Update last position for continuity
        if best_match:
            self.last_position = best_match
        
        return best_match
    
    def map_to_guitar(self, frequencies, confidences, times, onset_times):
        """
        Map detected frequencies to guitar tab notation
        
        Args:
            frequencies: Array of detected frequencies
            confidences: Confidence values for each frequency
            times: Time stamps for each frequency
            onset_times: Detected note onset times
            
        Returns:
            List of note dictionaries with timing and position info
        """
        notes = []
        self.last_position = None  # Reset position tracking
        
        # Group frequencies by onset times
        for i, onset_time in enumerate(onset_times):
            # Find frequencies near this onset
            if i < len(onset_times) - 1:
                next_onset = onset_times[i + 1]
            else:
                next_onset = times[-1] if len(times) > 0 else onset_time + 1.0
            
            # Get frequencies in this time window
            mask = (times >= onset_time) & (times < next_onset)
            if not np.any(mask):
                continue
            
            window_freqs = frequencies[mask]
            window_confs = confidences[mask]
            
            # Use the most confident frequency in this window
            if len(window_freqs) > 0:
                best_idx = np.argmax(window_confs)
                freq = window_freqs[best_idx]
                
                # Map to guitar position
                position = self._find_best_position(freq)
                
                if position and position['fret'] <= NUM_FRETS:
                    notes.append({
                        'time': onset_time,
                        'duration': next_onset - onset_time,
                        'string': position['string'],
                        'fret': position['fret'],
                        'frequency': freq,
                        'confidence': window_confs[best_idx]
                    })
        
        return notes
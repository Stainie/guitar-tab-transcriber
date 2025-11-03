"""
Advanced position selection for playability - Phase 3.4
"""
from typing import List, Dict, Optional
import numpy as np


class PositionOptimizer:
    """Optimize note positions for playability"""
    
    def __init__(self):
        self.max_fret_stretch = 4  # Maximum comfortable fret span
        self.max_string_jump = 3   # Maximum comfortable string jump
    
    def optimize_positions(self, notes: List[Dict]) -> List[Dict]:
        """
        Optimize note positions for natural playability
        
        Args:
            notes: List of fused notes
            
        Returns:
            Notes with optimized positions
        """
        if not notes:
            return notes
        
        optimized = []
        last_position = None
        
        for note in notes:
            # If note has high confidence and video confirmation, keep it
            if note.get('source') in ['fused', 'video_only'] and note.get('confidence', 0) > 0.8:
                optimized.append(note)
                last_position = {'string': note.get('string'), 'fret': note.get('fret')}
                continue
            
            # Otherwise, optimize position
            optimized_note = self._optimize_single_position(note, last_position)
            optimized.append(optimized_note)
            last_position = {'string': optimized_note.get('string'), 'fret': optimized_note.get('fret')}
        
        return optimized
    
    def _optimize_single_position(self, note: Dict, last_position: Optional[Dict]) -> Dict:
        """Optimize a single note's position"""
        # If no alternative positions, return as-is
        if 'video_position' not in note and 'alternate_fingering' not in note:
            return note
        
        current_string = note.get('string')
        current_fret = note.get('fret')
        
        # If we have video position as alternative, consider it
        if 'video_position' in note:
            video_pos = note['video_position']
            
            # Calculate playability score for both positions
            if last_position:
                current_score = self._playability_score(
                    current_string, current_fret,
                    last_position['string'], last_position['fret']
                )
                
                video_score = self._playability_score(
                    video_pos['string'], video_pos['fret'],
                    last_position['string'], last_position['fret']
                )
                
                # If video position is more playable, use it
                if video_score > current_score * 1.2:  # 20% better
                    note['string'] = video_pos['string']
                    note['fret'] = video_pos['fret']
                    note['position_optimized'] = True
        
        return note
    
    def _playability_score(self, string1: int, fret1: int, string2: int, fret2: int) -> float:
        """
        Score playability of transitioning from one position to another
        
        Higher score = more playable
        """
        score = 100.0
        
        # Penalize large fret jumps
        fret_jump = abs(fret1 - fret2)
        if fret_jump > self.max_fret_stretch:
            score -= (fret_jump - self.max_fret_stretch) * 10
        
        # Penalize large string jumps
        string_jump = abs(string1 - string2)
        if string_jump > self.max_string_jump:
            score -= (string_jump - self.max_string_jump) * 5
        
        # Prefer staying in same position area
        if fret_jump <= 2 and string_jump <= 1:
            score += 20
        
        return max(0, score)
    
    def group_into_chords(self, notes: List[Dict], time_window: float = 0.05) -> List[Dict]:
        """
        Group notes that occur simultaneously into chords
        
        Args:
            notes: List of notes
            time_window: Time window for grouping (seconds)
            
        Returns:
            List of note events (single notes or chords)
        """
        if not notes:
            return []
        
        # Sort by time
        sorted_notes = sorted(notes, key=lambda n: n.get('time', 0))
        
        grouped_events = []
        current_group = [sorted_notes[0]]
        current_time = sorted_notes[0].get('time', 0)
        
        for note in sorted_notes[1:]:
            note_time = note.get('time', 0)
            
            # If within time window, add to current group
            if abs(note_time - current_time) < time_window:
                current_group.append(note)
            else:
                # Finalize current group
                if len(current_group) > 1:
                    # It's a chord
                    grouped_events.append({
                        'type': 'chord',
                        'notes': current_group,
                        'time': current_time,
                        'num_notes': len(current_group)
                    })
                else:
                    # Single note
                    grouped_events.append({
                        'type': 'note',
                        'note': current_group[0],
                        'time': current_time
                    })
                
                # Start new group
                current_group = [note]
                current_time = note_time
        
        # Don't forget last group
        if len(current_group) > 1:
            grouped_events.append({
                'type': 'chord',
                'notes': current_group,
                'time': current_time,
                'num_notes': len(current_group)
            })
        elif current_group:
            grouped_events.append({
                'type': 'note',
                'note': current_group[0],
                'time': current_time
            })
        
        return grouped_events
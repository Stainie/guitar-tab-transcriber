"""
Multimodal fusion - combine audio and video predictions - Phase 3.2 & 3.3
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.fusion.confidence_scorer import ConfidenceScorer


class MultimodalFusion:
    """Fuse audio and video predictions intelligently"""
    
    def __init__(self, 
                 audio_weight=0.4,
                 video_weight=0.6):
        """
        Args:
            audio_weight: Weight for audio predictions (0-1)
            video_weight: Weight for video predictions (0-1)
        """
        self.audio_weight = audio_weight
        self.video_weight = video_weight
        self.confidence_scorer = ConfidenceScorer()
        
        # Timing tolerance for matching audio/video events (seconds)
        self.time_tolerance = 0.15
    
    def fuse_predictions(self,
                        audio_notes: List[Dict],
                        video_notes: List[Dict],
                        audio_context: Dict = None,
                        video_context: Dict = None) -> List[Dict]:
        """
        Fuse audio and video predictions into final transcription
        
        Args:
            audio_notes: Notes from audio analysis (Phase 1)
            video_notes: Notes from video analysis (Phase 2)
            audio_context: Additional audio context
            video_context: Additional video context
            
        Returns:
            List of fused note predictions
        """
        audio_context = audio_context or {}
        video_context = video_context or {}
        
        # Match audio and video notes by time
        matched_pairs, unmatched_audio, unmatched_video = self._match_notes_by_time(
            audio_notes, video_notes
        )
        
        fused_notes = []
        
        # Process matched pairs
        for audio_note, video_note in matched_pairs:
            fused_note = self._fuse_note_pair(
                audio_note, video_note,
                audio_context, video_context
            )
            if fused_note:
                fused_notes.append(fused_note)
        
        # Process unmatched audio notes (no video confirmation)
        for audio_note in unmatched_audio:
            audio_conf = self.confidence_scorer.score_audio_prediction(
                audio_note, audio_context
            )
            
            # Only include if confidence is high enough
            if audio_conf > 0.5:
                note = audio_note.copy()
                note['source'] = 'audio_only'
                note['confidence'] = audio_conf * self.audio_weight
                fused_notes.append(note)
        
        # Process unmatched video notes (no audio confirmation)
        for video_note in unmatched_video:
            video_conf = self.confidence_scorer.score_video_prediction(
                video_note, video_context
            )
            
            # Only include if confidence is high enough AND picking was detected
            if video_conf > 0.4 and video_note.get('played', False):
                note = video_note.copy()
                note['source'] = 'video_only'
                note['confidence'] = video_conf * self.video_weight
                fused_notes.append(note)
        
        # Sort by time
        fused_notes.sort(key=lambda n: n.get('time', 0))
        
        return fused_notes
    
    def _match_notes_by_time(self,
                            audio_notes: List[Dict],
                            video_notes: List[Dict]) -> Tuple[List, List, List]:
        """
        Match audio and video notes that occur at similar times
        
        Returns:
            Tuple of (matched_pairs, unmatched_audio, unmatched_video)
        """
        matched_pairs = []
        used_audio = set()
        used_video = set()
        
        # Try to match each video note with audio notes
        for v_idx, video_note in enumerate(video_notes):
            video_time = video_note.get('time', video_note.get('timestamp', 0))
            
            best_audio_idx = None
            min_time_diff = float('inf')
            
            # Find closest audio note in time
            for a_idx, audio_note in enumerate(audio_notes):
                if a_idx in used_audio:
                    continue
                
                audio_time = audio_note.get('time', 0)
                time_diff = abs(audio_time - video_time)
                
                if time_diff < self.time_tolerance and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_audio_idx = a_idx
            
            # If match found, pair them
            if best_audio_idx is not None:
                matched_pairs.append((audio_notes[best_audio_idx], video_note))
                used_audio.add(best_audio_idx)
                used_video.add(v_idx)
        
        # Collect unmatched notes
        unmatched_audio = [n for i, n in enumerate(audio_notes) if i not in used_audio]
        unmatched_video = [n for i, n in enumerate(video_notes) if i not in used_video]
        
        return matched_pairs, unmatched_audio, unmatched_video
    
    def _fuse_note_pair(self,
                       audio_note: Dict,
                       video_note: Dict,
                       audio_context: Dict,
                       video_context: Dict) -> Optional[Dict]:
        """
        Fuse a matched audio-video note pair
        
        Returns:
            Fused note or None if conflict cannot be resolved
        """
        # Score both predictions
        audio_conf = self.confidence_scorer.score_audio_prediction(audio_note, audio_context)
        video_conf = self.confidence_scorer.score_video_prediction(video_note, video_context)
        
        # Compare predictions
        comparison = self.confidence_scorer.compare_predictions(audio_note, video_note)
        
        # Case 1: Perfect agreement
        if comparison['string_match'] and comparison['fret_match']:
            return self._merge_agreeing_notes(
                audio_note, video_note, audio_conf, video_conf
            )
        
        # Case 2: Octave error (common in pitch detection)
        if comparison['frequency_diff_cents'] > 1100 and comparison['frequency_diff_cents'] < 1300:
            # Likely octave error - trust video more
            return self._resolve_octave_error(
                audio_note, video_note, audio_conf, video_conf
            )
        
        # Case 3: Close frequency but different position (alternate fingering)
        if comparison['frequency_diff_cents'] < 50:
            # Same note, different position - trust video for position
            return self._resolve_position_conflict(
                audio_note, video_note, audio_conf, video_conf
            )
        
        # Case 4: Significant disagreement
        if comparison['agreement'] < 0.5:
            # Use confidence-weighted voting
            if video_conf > audio_conf * 1.5:  # Video much more confident
                note = video_note.copy()
                note['confidence'] = video_conf * self.video_weight
                note['source'] = 'video_priority'
                return note
            elif audio_conf > video_conf * 1.5:  # Audio much more confident
                note = audio_note.copy()
                note['confidence'] = audio_conf * self.audio_weight
                note['source'] = 'audio_priority'
                # But use video position if available
                if 'string' in video_note and 'fret' in video_note:
                    note['string_video'] = video_note['string']
                    note['fret_video'] = video_note['fret']
                return note
            else:
                # Confidences similar but predictions differ - use weighted average
                return self._weighted_fusion(
                    audio_note, video_note, audio_conf, video_conf
                )
        
        # Default: merge with agreement weighting
        return self._merge_agreeing_notes(
            audio_note, video_note, audio_conf, video_conf, comparison['agreement']
        )
    
    def _merge_agreeing_notes(self,
                             audio_note: Dict,
                             video_note: Dict,
                             audio_conf: float,
                             video_conf: float,
                             agreement: float = 1.0) -> Dict:
        """Merge notes that agree (or mostly agree)"""
        # Weighted confidence
        total_weight = self.audio_weight * audio_conf + self.video_weight * video_conf
        
        fused_note = {
            'time': video_note.get('time', video_note.get('timestamp', audio_note.get('time', 0))),
            'string': video_note.get('string', audio_note.get('string')),
            'fret': video_note.get('fret', audio_note.get('fret')),
            'frequency': audio_note.get('frequency'),
            'confidence': total_weight * agreement,
            'source': 'fused',
            'audio_confidence': audio_conf,
            'video_confidence': video_conf,
            'agreement': agreement
        }
        
        # Include additional info
        if 'duration' in audio_note:
            fused_note['duration'] = audio_note['duration']
        
        if 'finger' in video_note:
            fused_note['finger'] = video_note['finger']
        
        if video_note.get('strumming'):
            fused_note['strumming'] = True
        
        return fused_note
    
    def _resolve_octave_error(self,
                             audio_note: Dict,
                             video_note: Dict,
                             audio_conf: float,
                             video_conf: float) -> Dict:
        """Resolve octave error by trusting video position"""
        note = video_note.copy()
        note['time'] = audio_note.get('time', video_note.get('time', video_note.get('timestamp', 0)))
        note['confidence'] = (video_conf * self.video_weight * 1.2)  # Boost for catching error
        note['source'] = 'octave_corrected'
        note['audio_frequency_raw'] = audio_note.get('frequency')
        note['correction'] = 'octave_error_fixed'
        
        if 'duration' in audio_note:
            note['duration'] = audio_note['duration']
        
        return note
    
    def _resolve_position_conflict(self,
                                  audio_note: Dict,
                                  video_note: Dict,
                                  audio_conf: float,
                                  video_conf: float) -> Dict:
        """Resolve position conflict (same note, different fingering)"""
        # Same pitch, trust video for exact position
        note = video_note.copy()
        note['time'] = audio_note.get('time', video_note.get('time', video_note.get('timestamp', 0)))
        note['frequency'] = audio_note.get('frequency')  # Use measured frequency
        note['confidence'] = (audio_conf * self.audio_weight + video_conf * self.video_weight)
        note['source'] = 'position_corrected'
        note['alternate_fingering'] = True
        
        if 'duration' in audio_note:
            note['duration'] = audio_note['duration']
        
        return note
    
    def _weighted_fusion(self,
                        audio_note: Dict,
                        video_note: Dict,
                        audio_conf: float,
                        video_conf: float) -> Dict:
        """Fuse conflicting predictions using weighted voting"""
        # Normalize weights
        total_conf = audio_conf * self.audio_weight + video_conf * self.video_weight
        audio_norm = (audio_conf * self.audio_weight) / total_conf
        video_norm = (video_conf * self.video_weight) / total_conf
        
        # Decide which prediction to use based on normalized confidence
        if video_norm > 0.6:  # Video has strong majority
            note = video_note.copy()
            note['source'] = 'video_weighted'
        else:  # Audio wins or close call
            note = audio_note.copy()
            note['source'] = 'audio_weighted'
            # But add video position as alternative
            note['video_position'] = {
                'string': video_note.get('string'),
                'fret': video_note.get('fret')
            }
        
        note['confidence'] = total_conf * 0.8  # Penalty for disagreement
        note['audio_confidence'] = audio_conf
        note['video_confidence'] = video_conf
        note['conflict_resolved'] = True
        
        return note
    
    def get_fusion_stats(self, fused_notes: List[Dict]) -> Dict:
        """Get statistics about the fusion process"""
        if not fused_notes:
            return {}
        
        sources = {}
        for note in fused_notes:
            source = note.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        avg_confidence = np.mean([n.get('confidence', 0) for n in fused_notes])
        
        corrections = sum(1 for n in fused_notes if 'correction' in n)
        conflicts = sum(1 for n in fused_notes if n.get('conflict_resolved', False))
        
        return {
            'total_notes': len(fused_notes),
            'source_distribution': sources,
            'average_confidence': avg_confidence,
            'corrections_made': corrections,
            'conflicts_resolved': conflicts
        }
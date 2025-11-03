"""
Generate ASCII guitar tablature - Enhanced for Phase 3
"""
from config.guitar_config import NUM_STRINGS
import numpy as np


class TabGenerator:
    """Generate ASCII tab format from note data"""
    
    def __init__(self, chars_per_second=8):
        """
        Args:
            chars_per_second: Character density in tab output
        """
        self.chars_per_second = chars_per_second
    
    def generate(self, notes: list) -> str:
        """
        Generate ASCII tab from notes
        
        Args:
            notes: List of note dictionaries with timing and position
            
        Returns:
            String containing ASCII guitar tab
        """
        if not notes:
            return "No notes detected in audio/video."
        
        # Filter out any invalid notes
        valid_notes = []
        for note in notes:
            string = note.get('string')
            fret = note.get('fret')
            time = note.get('time', 0)
            
            if string is not None and fret is not None and string < NUM_STRINGS:
                valid_notes.append(note)
        
        if not valid_notes:
            return "No valid notes detected (all notes filtered out)."
        
        print(f"\n[Tab Generator] Processing {len(valid_notes)} valid notes...")
        
        # Calculate tab length based on last note
        max_time = max(note.get('time', 0) + note.get('duration', 0.1) 
                      for note in valid_notes)
        tab_length = int(max_time * self.chars_per_second) + 20
        
        # Initialize tab lines (one per string)
        tab_lines = []
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']  # High to low
        
        for i in range(NUM_STRINGS):
            tab_lines.append(['-'] * tab_length)
        
        # Track positions to avoid overlaps
        occupied = [set() for _ in range(NUM_STRINGS)]
        
        # Place notes on the tab
        notes_placed = 0
        for note in valid_notes:
            position = int(note.get('time', 0) * self.chars_per_second)
            
            if position >= tab_length:
                continue
            
            string_idx = NUM_STRINGS - 1 - note.get('string')  # Reverse for display
            fret = note.get('fret')
            fret_str = str(fret)
            
            # Check if position is available (not too close to another note)
            available = True
            for offset in range(len(fret_str)):
                if position + offset in occupied[string_idx]:
                    available = False
                    break
            
            if not available:
                # Try next position
                position += 1
                if position >= tab_length:
                    continue
            
            # Place the note
            for j, char in enumerate(fret_str):
                if position + j < tab_length:
                    tab_lines[string_idx][position + j] = char
                    occupied[string_idx].add(position + j)
            
            notes_placed += 1
        
        print(f"[Tab Generator] Placed {notes_placed} notes on tab")
        
        # Build output string
        output = []
        output.append("=" * 70)
        output.append("GUITAR TAB - Multimodal Transcription")
        output.append("=" * 70)
        output.append("")
        
        # Add metadata if available
        if valid_notes and 'source' in valid_notes[0]:
            sources = {}
            for note in valid_notes:
                src = note.get('source', 'unknown')
                sources[src] = sources.get(src, 0) + 1
            
            output.append("Note sources:")
            for src, count in sorted(sources.items()):
                output.append(f"  {src}: {count}")
            output.append("")
        
        # Add the tab
        for i, line in enumerate(tab_lines):
            string_name = string_names[i]
            tab_line = f"{string_name}|{''.join(line)}"
            output.append(tab_line)
        
        output.append("")
        
        # Statistics
        output.append(f"Total notes: {len(valid_notes)}")
        output.append(f"Notes placed: {notes_placed}")
        output.append(f"Duration: {max_time:.2f} seconds")
        
        # Confidence statistics if available
        confidences = [n.get('confidence') for n in valid_notes if 'confidence' in n]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            output.append(f"Average confidence: {avg_conf:.2f}")
        
        output.append("=" * 70)
        
        return '\n'.join(output)
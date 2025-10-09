"""
Generate ASCII guitar tablature
"""
from config.guitar_config import NUM_STRINGS


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
            return "No notes detected in audio."
        
        # Calculate tab length based on last note
        max_time = max(note['time'] + note['duration'] for note in notes)
        tab_length = int(max_time * self.chars_per_second) + 10
        
        # Initialize tab lines (one per string)
        tab_lines = []
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']  # High to low
        
        for i in range(NUM_STRINGS):
            tab_lines.append(['-'] * tab_length)
        
        # Place notes on the tab
        for note in notes:
            position = int(note['time'] * self.chars_per_second)
            if position < tab_length:
                string_idx = NUM_STRINGS - 1 - note['string']  # Reverse for display
                fret_str = str(note['fret'])
                
                # Clear dashes for the fret number
                for j in range(len(fret_str)):
                    if position + j < tab_length:
                        tab_lines[string_idx][position + j] = fret_str[j] if j == 0 else fret_str[j]
        
        # Build output string
        output = []
        output.append("=" * 60)
        output.append("GUITAR TAB - Phase 1 Transcription")
        output.append("=" * 60)
        output.append("")
        
        # Add the tab
        for i, line in enumerate(tab_lines):
            string_name = string_names[i]
            tab_line = f"{string_name}|{''.join(line)}"
            output.append(tab_line)
        
        output.append("")
        output.append(f"Total notes: {len(notes)}")
        output.append(f"Duration: {max_time:.2f} seconds")
        output.append("=" * 60)
        
        return '\n'.join(output)
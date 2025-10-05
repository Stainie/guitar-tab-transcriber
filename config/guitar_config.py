# Standard guitar tuning (E A D G B E)
STANDARD_TUNING = [
    ('E', 2),  # Low E  - 82.41 Hz
    ('A', 2),  # A      - 110.00 Hz
    ('D', 3),  # D      - 146.83 Hz
    ('G', 3),  # G      - 196.00 Hz
    ('B', 3),  # B      - 246.94 Hz
    ('E', 4),  # High E - 329.63 Hz
]

NUM_FRETS = 24 # Should be switchable to 22 (or 21)
NUM_STRINGS = 6 # Standard 6-string guitar, should be switchable to 7, 8, etc. (perhaps 4/5 if we want to support bass, or Jacob Collier's dubmass guitar)

# Frequency tolerance for pitch matching (in cents)
PITCH_TOLERANCE = 50  # 50 cents = half a semitone
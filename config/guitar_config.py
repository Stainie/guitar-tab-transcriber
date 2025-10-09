# Standard guitar tuning (E A D G B E)
STANDARD_TUNING = [
    ('E', 2),  # Low E  - 82.41 Hz
    ('A', 2),  # A      - 110.00 Hz
    ('D', 3),  # D      - 146.83 Hz
    ('G', 3),  # G      - 196.00 Hz
    ('B', 3),  # B      - 246.94 Hz
    ('E', 4),  # High E - 329.63 Hz
]

# 1 step down tuning (D G C F A D)
ONE_STEP_DOWN_TUNING = [
    ('D', 2),  # Low D  - 73.42 Hz
    ('G', 2),  # G      - 98.00 Hz
    ('C', 3),  # C      - 130.81 Hz
    ('F', 3),  # F      - 174.61 Hz
    ('A', 3),  # A      - 220.00 Hz
    ('D', 4),  # High D - 293.66 Hz
]

NUM_FRETS = 22 # Should be switchable to 22 (or 21)
NUM_STRINGS = 6 # Standard 6-string guitar, should be switchable to 7, 8, etc. (perhaps 4/5 if we want to support bass, or Jacob Collier's dumbass guitar)

# Frequency tolerance for pitch matching (in cents)
PITCH_TOLERANCE = 50  # 50 cents = half a semitone

# Prefer lower fret positions (more playable)
FRET_PENALTY_WEIGHT = 0.5  # Penalty increases with fret number
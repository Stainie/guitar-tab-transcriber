# Standard guitar tuning (E A D G B E)
STANDARD_TUNING = [
    ('E', 2),  # Low E  - 82.41 Hz
    ('A', 2),  # A      - 110.00 Hz
    ('D', 3),  # D      - 146.83 Hz
    ('G', 3),  # G      - 196.00 Hz
    ('B', 3),  # B      - 246.94 Hz
    ('E', 4),  # High E - 329.63 Hz
]

NUM_FRETS = 22
NUM_STRINGS = 6

# Frequency tolerance for pitch matching (in cents)
PITCH_TOLERANCE = 50  # 50 cents = half a semitone

# Prefer lower fret positions (more playable)
FRET_PENALTY_WEIGHT = 0.5  # Penalty increases with fret number

# Phase 2: Video Analysis Config
# Fretboard detection
FRETBOARD_MIN_AREA_RATIO = 0.15  # Minimum 15% of frame
FRETBOARD_MAX_AREA_RATIO = 0.80  # Maximum 80% of frame
FRETBOARD_ASPECT_RATIO_MIN = 1.5  # Width should be > 1.5x height
FRETBOARD_ASPECT_RATIO_MAX = 6.0  # Width should be < 6x height

# String detection
STRING_DETECTION_SENSITIVITY = 30  # Hough transform threshold
STRING_MIN_LENGTH_RATIO = 0.3  # Minimum 30% of fretboard width

# Calibration
CALIBRATION_FRAMES_REQUIRED = 5  # Number of frames needed for calibration
CALIBRATION_CONSISTENCY_THRESHOLD = 0.8  # 80% agreement required
#!/usr/bin/env python3
import sys

def check_imports():
    """Verify all required packages are installed"""
    packages = ['librosa', 'crepe', 'aubio', 'music21', 'numpy', 'scipy']
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError as e:
            print(f"✗ {pkg} - {e}")
            return False
    return True

def check_audio_file():
    """Check if we can process a simple audio file"""
    import librosa
    import numpy as np
    
    # Generate a simple sine wave (A4 = 440 Hz)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # Try to process it
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    print(f"✓ Audio processing works")
    return True

if __name__ == "__main__":
    print("Validating setup...\n")
    
    if check_imports():
        print("\n✓ All packages installed")
    else:
        print("\n✗ Some packages missing")
        sys.exit(1)
    
    if check_audio_file():
        print("✓ Audio processing functional")
    
    print("\n✓ Setup complete! Ready to build.")
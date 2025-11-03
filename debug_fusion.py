#!/usr/bin/env python3
"""
Debug script to see what's happening at each stage
"""
import sys
from pathlib import Path
from src.pipeline import TranscriptionPipeline


def debug_pipeline(video_path: str):
    """Debug each stage of the pipeline"""
    
    video_path = Path(video_path)
    
    print("=" * 80)
    print("DEBUGGING FUSION PIPELINE")
    print("=" * 80)
    
    # Run pipeline
    pipeline = TranscriptionPipeline(mode='multimodal')
    result = pipeline.process(str(video_path), output_debug=True)
    
    # Analyze each stage
    print("\n" + "=" * 80)
    print("STAGE-BY-STAGE ANALYSIS")
    print("=" * 80)
    
    # Audio notes
    print("\n[1] AUDIO NOTES:")
    if result['audio_notes']:
        print(f"  Total: {len(result['audio_notes'])}")
        print(f"  Sample (first 5):")
        for i, note in enumerate(result['audio_notes'][:5]):
            print(f"    {i+1}. Time: {note.get('time', 0):.2f}s, "
                  f"String: {note.get('string')}, Fret: {note.get('fret')}, "
                  f"Freq: {note.get('frequency', 0):.1f}Hz, "
                  f"Conf: {note.get('confidence', 0):.2f}")
    else:
        print("  None detected")
    
    # Video notes
    print("\n[2] VIDEO NOTES:")
    if result['video_notes']:
        print(f"  Total: {len(result['video_notes'])}")
        print(f"  Sample (first 5):")
        for i, note in enumerate(result['video_notes'][:5]):
            print(f"    {i+1}. Time: {note.get('time', note.get('timestamp', 0)):.2f}s, "
                  f"String: {note.get('string')}, Fret: {note.get('fret')}, "
                  f"Played: {note.get('played')}, Strum: {note.get('strumming')}")
    else:
        print("  None detected")
    
    # Fused notes
    print("\n[3] FUSED NOTES:")
    if result['fused_notes']:
        print(f"  Total: {len(result['fused_notes'])}")
        
        # Analyze by source
        sources = {}
        for note in result['fused_notes']:
            source = note.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"  Sources breakdown:")
        for source, count in sources.items():
            print(f"    {source}: {count}")
        
        print(f"\n  Sample (first 10):")
        for i, note in enumerate(result['fused_notes'][:10]):
            print(f"    {i+1}. Time: {note.get('time', 0):.2f}s, "
                  f"S{note.get('string')}F{note.get('fret')}, "
                  f"Source: {note.get('source')}, "
                  f"Conf: {note.get('confidence', 0):.2f}")
    else:
        print("  None after fusion!")
    
    # Analyze confidence distribution
    print("\n[4] CONFIDENCE ANALYSIS:")
    if result['fused_notes']:
        confidences = [n.get('confidence', 0) for n in result['fused_notes']]
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
        print(f"  Avg: {sum(confidences)/len(confidences):.3f}")
        
        # Count by confidence ranges
        ranges = {
            'Very High (>0.8)': sum(1 for c in confidences if c > 0.8),
            'High (0.6-0.8)': sum(1 for c in confidences if 0.6 <= c <= 0.8),
            'Medium (0.4-0.6)': sum(1 for c in confidences if 0.4 <= c <= 0.6),
            'Low (<0.4)': sum(1 for c in confidences if c < 0.4),
        }
        print(f"  Distribution:")
        for range_name, count in ranges.items():
            print(f"    {range_name}: {count}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    audio_count = len(result['audio_notes']) if result['audio_notes'] else 0
    video_count = len(result['video_notes']) if result['video_notes'] else 0
    fused_count = len(result['fused_notes']) if result['fused_notes'] else 0
    
    if fused_count < audio_count * 0.5:
        print("\n⚠️  WARNING: Fusion dropped more than 50% of audio notes!")
        print("   Possible causes:")
        print("   - Confidence thresholds too high")
        print("   - Time matching window too narrow")
        print("   - Video detection poor (check debug frames)")
    
    if fused_count < video_count * 0.3:
        print("\n⚠️  WARNING: Fusion dropped most video notes!")
        print("   Possible causes:")
        print("   - Video notes not being played (picking not detected)")
        print("   - Confidence thresholds too strict")
    
    if audio_count > 0 and video_count == 0:
        print("\n⚠️  ERROR: No video notes detected!")
        print("   Check:")
        print("   - Are hands visible in frame?")
        print("   - Is fretboard calibrated? (check debug frames)")
        print("   - Is picking motion visible?")
    
    if fused_count > 0:
        print(f"\n✓ Pipeline working, but may need tuning")
        print(f"  Consider adjusting confidence thresholds if output is too sparse")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_fusion.py <video_file>")
        sys.exit(1)
    
    debug_pipeline(sys.argv[1])
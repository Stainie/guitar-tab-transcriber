#!/usr/bin/env python3
"""
Test script for Phase 3: Complete multimodal fusion pipeline
"""
import sys
from pathlib import Path
import json

from src.pipeline import TranscriptionPipeline


def test_multimodal_fusion(video_path: str, 
                           mode: str = 'multimodal',
                           save_debug: bool = True):
    """
    Test complete Phase 3 pipeline with all modes
    
    Args:
        video_path: Path to input video
        mode: 'audio', 'video', or 'multimodal'
        save_debug: Whether to save debug visualizations
    """
    
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("=" * 80)
    print("PHASE 3: MULTIMODAL FUSION TEST")
    print("=" * 80)
    print(f"Input: {video_path}")
    print(f"Mode: {mode}")
    print(f"Debug output: {save_debug}")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = TranscriptionPipeline(
        mode=mode,
        audio_weight=0.4,
        video_weight=0.6
    )
    
    # Process
    result = pipeline.process(str(video_path), output_debug=save_debug)
    
    # Save outputs
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    base_name = video_path.stem
    
    # Save tab
    tab_path = output_dir / f"{base_name}_{mode}_tab.txt"
    with open(tab_path, 'w') as f:
        f.write(result['tab'])
    
    print(f"\n{'='*80}")
    print("OUTPUT FILES")
    print(f"{'='*80}")
    print(f"Tab saved to: {tab_path}")
    
    # Save detailed results as JSON
    json_path = output_dir / f"{base_name}_{mode}_results.json"
    
    # Prepare JSON-serializable data
    json_data = {
        'input_file': str(video_path),
        'mode': mode,
        'metadata': result['metadata'],
        'note_count': {
            'audio': len(result['audio_notes']) if result['audio_notes'] else 0,
            'video': len(result['video_notes']) if result['video_notes'] else 0,
            'fused': len(result['fused_notes']) if result['fused_notes'] else 0
        }
    }
    
    # Add sample notes for inspection
    if result['fused_notes']:
        json_data['sample_notes'] = [
            {
                'time': n.get('time', 0),
                'string': n.get('string'),
                'fret': n.get('fret'),
                'confidence': n.get('confidence'),
                'source': n.get('source')
            }
            for n in result['fused_notes'][:20]  # First 20 notes
        ]
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if mode == 'multimodal' and result['fused_notes']:
        fusion_stats = result['metadata'].get('fusion_stats', {})
        
        print(f"\nNotes detected:")
        print(f"  Audio only: {json_data['note_count']['audio']}")
        print(f"  Video only: {json_data['note_count']['video']}")
        print(f"  After fusion: {json_data['note_count']['fused']}")
        
        print(f"\nFusion quality:")
        print(f"  Average confidence: {fusion_stats.get('average_confidence', 0):.2f}")
        print(f"  Corrections made: {fusion_stats.get('corrections_made', 0)}")
        print(f"  Conflicts resolved: {fusion_stats.get('conflicts_resolved', 0)}")
        
        print(f"\nNote sources:")
        for source, count in fusion_stats.get('sources', {}).items():
            pct = count / fusion_stats.get('total_notes', 1) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE!")
    print(f"{'='*80}")
    print("\nReview the outputs:")
    print(f"  1. Tab file: {tab_path}")
    print(f"  2. Results JSON: {json_path}")
    if save_debug:
        print(f"  3. Debug frames: data/debug/phase3_fusion/")
    
    print("\nNext steps:")
    print("  - Compare audio-only, video-only, and multimodal results")
    print("  - Check fusion statistics for accuracy insights")
    print("  - Review debug frames to verify detection quality")


def compare_modes(video_path: str):
    """Run all three modes and compare results"""
    
    video_path = Path(video_path)
    
    print("=" * 80)
    print("COMPARING ALL MODES")
    print("=" * 80)
    
    results = {}
    
    for mode in ['audio', 'video', 'multimodal']:
        print(f"\n{'='*80}")
        print(f"Testing mode: {mode.upper()}")
        print(f"{'='*80}\n")
        
        pipeline = TranscriptionPipeline(mode=mode)
        result = pipeline.process(str(video_path), output_debug=False)
        
        # Count notes
        if mode == 'audio':
            note_count = len(result['audio_notes'])
        elif mode == 'video':
            note_count = len(result['video_notes'])
        else:
            note_count = len(result['fused_notes'])
        
        results[mode] = {
            'note_count': note_count,
            'tab': result['tab']
        }
        
        print(f"\n{mode.upper()} mode: {note_count} notes detected\n")
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print("Note counts:")
    for mode, data in results.items():
        print(f"  {mode:12s}: {data['note_count']:4d} notes")
    
    print("\nTabs saved to data/output/ with suffixes:")
    print("  - _audio_tab.txt")
    print("  - _video_tab.txt")
    print("  - _multimodal_tab.txt")
    
    print("\nRecommendation:")
    if results['multimodal']['note_count'] >= results['audio']['note_count'] * 0.8:
        print("  ✓ Multimodal fusion is working well!")
    else:
        print("  ⚠ Multimodal detected significantly fewer notes - check video quality")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_phase3_fusion.py <video_file> [mode] [--compare]")
        print("\nModes:")
        print("  audio      - Audio analysis only (Phase 1)")
        print("  video      - Video analysis only (Phase 2)")
        print("  multimodal - Combined analysis (Phase 3) [DEFAULT]")
        print("\nOptions:")
        print("  --compare  - Run all three modes and compare results")
        print("\nExamples:")
        print("  python test_phase3_fusion.py data/input/guitar.mp4")
        print("  python test_phase3_fusion.py data/input/guitar.mp4 audio")
        print("  python test_phase3_fusion.py data/input/guitar.mp4 --compare")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    if '--compare' in sys.argv:
        compare_modes(video_file)
    else:
        mode = sys.argv[2] if len(sys.argv) > 2 else 'multimodal'
        if mode not in ['audio', 'video', 'multimodal']:
            print(f"Error: Invalid mode '{mode}'. Use: audio, video, or multimodal")
            sys.exit(1)
        
        test_multimodal_fusion(video_file, mode=mode, save_debug=True)
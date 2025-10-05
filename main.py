#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.pipeline import TranscriptionPipeline

def main():
    parser = argparse.ArgumentParser(description='Guitar Tab Transcriber - Phase 1')
    parser.add_argument('input', type=str, help='Input audio or video file')
    parser.add_argument('-o', '--output', type=str, help='Output tab file (default: auto-generated)')
    parser.add_argument('--format', choices=['txt', 'ascii'], default='txt', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TranscriptionPipeline()
    
    # Process
    print(f"Processing: {args.input}")
    tab = pipeline.process(args.input)
    
    # Save output
    output_path = args.output or f"data/output/{Path(args.input).stem}_tab.txt"
    with open(output_path, 'w') as f:
        f.write(tab)
    
    print(f"Tab saved to: {output_path}")

if __name__ == "__main__":
    main()
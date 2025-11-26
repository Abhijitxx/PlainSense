"""
Batch Process Medical Lab Reports - IMPROVED VERSION
====================================================

This script processes all 426 lab reports with optimized settings:
- ML correction DISABLED (prevents medical term degradation)
- Medical-specific corrections ENABLED
- Tesseract PSM 6 (uniform block of text)

Expected accuracy: 85-90% on medical reports
Processing time: ~71 minutes for 426 images
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Suppress encoding warnings
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

from pipeline import CompletePipeline

def batch_process_reports():
    """Process all lab reports with improved settings"""
    
    # Configuration
    input_dir = Path("Dataset/Reports")
    output_dir = Path("output/final_preprocessed")
    
    print("="*70)
    print("BATCH PROCESSING: MEDICAL LAB REPORTS")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Settings: ML correction OFF, Medical corrections ON")
    print("="*70)
    
    # Get all PNG files
    png_files = list(input_dir.glob("*.png"))
    total_files = len(png_files)
    
    print(f"\nFound {total_files} PNG files to process")
    
    if total_files == 0:
        print("ERROR: No PNG files found!")
        return
    
    # Ask for confirmation
    response = input(f"\nProcess all {total_files} files? This will take ~{total_files * 10 / 60:.0f} minutes (y/n): ")
    if response.lower() != 'y':
        print("Cancelled by user")
        return
    
    # Initialize pipeline WITHOUT ML correction
    print("\nInitializing pipeline...")
    pipeline = CompletePipeline(use_ml_correction=False)
    
    # Process files
    start_time = time.time()
    successful = 0
    failed = 0
    total_words = 0
    
    print(f"\n{'='*70}")
    print("PROCESSING FILES")
    print(f"{'='*70}\n")
    
    for i, file_path in enumerate(png_files, 1):
        print(f"[{i}/{total_files}] Processing: {file_path.name[:50]}...")
        
        try:
            result = pipeline.process_document(str(file_path))
            
            if result['success']:
                successful += 1
                words = result.get('word_count', 0)
                total_words += words
                print(f"  -> SUCCESS ({words} words)")
            else:
                failed += 1
                print(f"  -> FAILED: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            failed += 1
            print(f"  -> ERROR: {e}")
        
        # Progress report every 50 files
        if i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_files - i) * avg_time
            print(f"\n  Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            print(f"  Elapsed: {elapsed/60:.1f} min, Remaining: {remaining/60:.1f} min")
            print(f"  Success: {successful}, Failed: {failed}\n")
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total files: {total_files}")
    print(f"Successful: {successful} ({successful/total_files*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_files*100:.1f}%)")
    print(f"Total words extracted: {total_words:,}")
    print(f"Average words per file: {total_words/successful if successful > 0 else 0:.0f}")
    print(f"Processing time: {total_time/60:.1f} minutes")
    print(f"Average time per file: {total_time/total_files:.1f} seconds")
    print(f"{'='*70}")
    
    # Save summary
    summary_file = output_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_file.write_text(
        f"Batch Processing Summary\n"
        f"{'='*70}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total files: {total_files}\n"
        f"Successful: {successful} ({successful/total_files*100:.1f}%)\n"
        f"Failed: {failed}\n"
        f"Total words: {total_words:,}\n"
        f"Processing time: {total_time/60:.1f} minutes\n"
        f"Settings: ML correction OFF, Medical corrections ON, Tesseract PSM 6\n"
    )
    print(f"\nSummary saved to: {summary_file}")

if __name__ == '__main__':
    batch_process_reports()

#!/usr/bin/env python3
"""
Quick test script to verify the watermarking framework works on a few sample files.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.utils.watermark_methods import get_all_methods, WatermarkMethod
from src.utils.metrics import evaluate_performance, compute_pesq
from src.audioio.audio_io import load_audio, save_audio, preprocess


def quick_test():
    """Run a quick test on a few sample files."""
    print("Quick Test of Watermarking Framework")
    print("=" * 50)
    
    # Configuration
    audio_dir = "100sample_wav"
    output_dir = "quick_test_results_updated"
    sample_rate = 16000
    max_files = 5  # Test only 5 files for quick verification
    
    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get methods
    methods = get_all_methods()
    print(f"Testing {len(methods)} methods: {[m.name for m in methods]}")
    
    # Get sample audio files
    audio_files = []
    speech_dir = Path(audio_dir) / "speech_wav"
    music_dir = Path(audio_dir) / "music_wav"
    
    if speech_dir.exists():
        audio_files.extend(list(speech_dir.glob("*.wav"))[:3])
    if music_dir.exists():
        audio_files.extend(list(music_dir.glob("*.wav"))[:2])
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Testing on {len(audio_files)} files:")
    for f in audio_files:
        print(f"  - {f.name}")
    print()
    
    # Generate watermark
    watermark_bits = [random.randint(0, 1) for _ in range(32)]
    print(f"Watermark: {''.join(map(str, watermark_bits[:16]))}... (32 bits)")
    print()
    
    results = []
    
    # Test each method on each file
    for method in methods:
        print(f"Testing {method.name}...")
        
        # Calibrate method
        try:
            # Use first file for calibration
            cal_audio = load_audio(str(audio_files[0]), sample_rate)
            cal_audio = preprocess(cal_audio)
            
            print(f"  Calibrating...")
            alpha = method.calibrate(cal_audio, sample_rate, target_pesq_drop=0.05)
            print(f"  Calibrated with alpha={alpha:.6f}")
            
        except Exception as e:
            print(f"  Calibration failed: {e}")
            continue
        
        # Test on all files
        for audio_file in audio_files:
            try:
                print(f"  Processing {audio_file.name}...")
                
                # Load and process audio
                audio = load_audio(str(audio_file), sample_rate)
                audio = preprocess(audio)
                
                # Embed watermark
                watermarked, metadata = method.embed(audio, sample_rate, watermark_bits)
                
                # Extract watermark
                extracted = method.extract(watermarked, sample_rate, len(watermark_bits), metadata)
                
                # Compute metrics
                metrics = evaluate_performance(audio, watermarked, extracted, watermark_bits, sample_rate)
                
                # Save watermarked audio
                output_filename = f"{audio_file.stem}_{method.name}_watermarked.wav"
                output_path = os.path.join(output_dir, output_filename)
                save_audio(watermarked, sample_rate, output_path)
                
                # Store results
                result = {
                    'file': audio_file.name,
                    'method': method.name,
                    'alpha': alpha,
                    **metrics
                }
                results.append(result)
                
                print(f"    ✓ SiSNR: {metrics['SiSNR']:.2f} dB, BER: {metrics['BER']:.4f}, TPR: {metrics['TPR']:.4f}, FPR: {metrics['FPR']:.4f}, PESQ: {metrics['PESQ']:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print()
    
    # Generate results
    if results:
        df = pd.DataFrame(results)
        results_file = os.path.join(output_dir, "quick_test_results.csv")
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
        # Print summary
        print("\nSummary:")
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            print(f"\n{method}:")
            print(f"  Files: {len(method_data)}")
            print(f"  Avg SiSNR: {method_data['SiSNR'].mean():.2f} dB")
            print(f"  Avg BER: {method_data['BER'].mean():.4f}")
            print(f"  Avg TPR: {method_data['TPR'].mean():.4f}")
            print(f"  Avg FPR: {method_data['FPR'].mean():.4f}")
            print(f"  Avg PESQ: {method_data['PESQ'].mean():.3f}")
    else:
        print("No successful tests completed.")


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducibility
    quick_test()

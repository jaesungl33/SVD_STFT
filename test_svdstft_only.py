#!/usr/bin/env python3
"""
Focused testing of SVD_STFT watermarking method only.
Provides comprehensive evaluation metrics for your method.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.utils.watermark_methods import SVDSTFTMethod
from src.utils.metrics import evaluate_performance, compute_pesq
from src.audioio.audio_io import load_audio, save_audio, preprocess


def test_svdstft_only():
    """Test only the SVD_STFT method on all 100 sample files."""
    print("SVD_STFT WATERMARKING METHOD EVALUATION")
    print("=" * 60)
    
    # Configuration
    audio_dir = "100sample_wav"
    output_dir = "svdstft_results_only"
    sample_rate = 16000
    target_pesq_drop = 0.05
    
    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "watermarked"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Initialize SVD_STFT method
    method = SVDSTFTMethod()
    print(f"Testing: {method.name}")
    print(f"Target PESQ drop: {target_pesq_drop}")
    print()
    
    # Get all audio files
    audio_files = []
    speech_dir = Path(audio_dir) / "speech_wav"
    music_dir = Path(audio_dir) / "music_wav"
    
    if speech_dir.exists():
        audio_files.extend(list(speech_dir.glob("*.wav")))
    if music_dir.exists():
        audio_files.extend(list(music_dir.glob("*.wav")))
    
    audio_files = sorted(audio_files)
    print(f"Found {len(audio_files)} audio files for testing")
    print()
    
    # Step 1: Calibrate the method
    print("STEP 1: CALIBRATING SVD_STFT METHOD")
    print("-" * 40)
    
    try:
        # Use first file for calibration
        cal_audio = load_audio(str(audio_files[0]), sample_rate)
        cal_audio = preprocess(cal_audio)
        
        print("Calibrating...")
        alpha = method.calibrate(cal_audio, sample_rate, target_pesq_drop)
        print(f"✓ Calibrated with alpha = {alpha:.6f}")
        
        # Verify calibration
        test_bits = [random.randint(0, 1) for _ in range(32)]
        watermarked, metadata = method.embed(cal_audio, sample_rate, test_bits)
        
        pesq_orig = compute_pesq(cal_audio, cal_audio, sample_rate)
        pesq_wm = compute_pesq(cal_audio, watermarked, sample_rate)
        pesq_drop = pesq_orig - pesq_wm
        
        print(f"✓ PESQ drop: {pesq_drop:.4f} (target: ≤{target_pesq_drop})")
        
    except Exception as e:
        print(f"✗ Calibration failed: {e}")
        return
    
    print()
    
    # Step 2: Test on all files
    print("STEP 2: TESTING ON ALL AUDIO FILES")
    print("-" * 40)
    
    # Generate watermark
    watermark_bits = [random.randint(0, 1) for _ in range(64)]
    print(f"Using watermark: {''.join(map(str, watermark_bits[:16]))}... (64 bits total)")
    print()
    
    results = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
            
            # Load and process audio
            audio = load_audio(str(audio_file), sample_rate)
            audio = preprocess(audio)
            
            # Embed watermark
            start_time = time.time()
            watermarked, metadata = method.embed(audio, sample_rate, watermark_bits)
            embed_time = time.time() - start_time
            
            # Extract watermark
            start_time = time.time()
            extracted = method.extract(watermarked, sample_rate, len(watermark_bits), metadata)
            extract_time = time.time() - start_time
            
            # Compute all evaluation metrics
            metrics = evaluate_performance(audio, watermarked, extracted, watermark_bits, sample_rate)
            
            # Save watermarked audio
            output_filename = f"{audio_file.stem}_SVD_STFT_watermarked.wav"
            output_path = os.path.join(output_dir, "watermarked", output_filename)
            save_audio(watermarked, sample_rate, output_path)
            
            # Store results
            result = {
                'file': audio_file.name,
                'file_size_mb': os.path.getsize(audio_file) / (1024 * 1024),
                'duration_seconds': len(audio) / sample_rate,
                'alpha': alpha,
                'embed_time': embed_time,
                'extract_time': extract_time,
                **metrics
            }
            results.append(result)
            
            print(f"  ✓ SiSNR: {metrics['SiSNR']:.2f} dB, BER: {metrics['BER']:.4f}, TPR: {metrics['TPR']:.4f}, FPR: {metrics['FPR']:.4f}, PESQ: {metrics['PESQ']:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nCompleted {len(results)} successful tests")
    print()
    
    # Step 3: Generate comprehensive results
    print("STEP 3: GENERATING COMPREHENSIVE RESULTS")
    print("-" * 40)
    
    if results:
        df = pd.DataFrame(results)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "results", "svdstft_comprehensive_results.csv")
        df.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")
        
        # Generate summary statistics
        summary = generate_summary_statistics(df)
        summary_file = os.path.join(output_dir, "results", "svdstft_summary_statistics.csv")
        summary.to_csv(summary_file, index=True)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Print comprehensive summary
        print_comprehensive_summary(df)
        
        # Save summary report
        save_summary_report(df, output_dir)
        
    else:
        print("No successful tests completed.")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics for SVD_STFT method."""
    stats = {
        'Metric': [
            'Files_Processed',
            'Avg_SiSNR_dB', 'Std_SiSNR_dB',
            'Avg_BER', 'Std_BER',
            'Avg_NC', 'Std_NC',
            'Avg_TPR', 'Std_TPR',
            'Avg_FPR', 'Std_FPR',
            'Avg_PESQ', 'Std_PESQ',
            'Avg_PSNR_dB', 'Std_PSNR_dB',
            'Avg_MSE', 'Std_MSE',
            'Avg_SSIM', 'Std_SSIM',
            'Avg_Embed_Time_s', 'Std_Embed_Time_s',
            'Avg_Extract_Time_s', 'Std_Extract_Time_s'
        ],
        'Value': [
            len(df),
            df['SiSNR'].mean(), df['SiSNR'].std(),
            df['BER'].mean(), df['BER'].std(),
            df['NC'].mean(), df['NC'].std(),
            df['TPR'].mean(), df['TPR'].std(),
            df['FPR'].mean(), df['FPR'].std(),
            df['PESQ'].mean(), df['PESQ'].std(),
            df['PSNR'].mean(), df['PSNR'].std(),
            df['MSE'].mean(), df['MSE'].std(),
            df['SSIM'].mean(), df['SSIM'].std(),
            df['embed_time'].mean(), df['embed_time'].std(),
            df['extract_time'].mean(), df['extract_time'].std()
        ]
    }
    
    return pd.DataFrame(stats)


def print_comprehensive_summary(df: pd.DataFrame):
    """Print comprehensive summary of SVD_STFT results."""
    print("\n" + "=" * 80)
    print("SVD_STFT COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n📊 OVERVIEW:")
    print(f"  Total files processed: {len(df)}")
    print(f"  Audio types: Music + Speech samples")
    print(f"  Watermark payload: 64 bits per file")
    print(f"  Calibration target: PESQ drop ≤ 0.05")
    
    print(f"\n🎯 WATERMARKING PERFORMANCE:")
    print(f"  Bit Error Rate (BER): {df['BER'].mean():.4f} ± {df['BER'].std():.4f}")
    print(f"  Normalized Correlation (NC): {df['NC'].mean():.4f} ± {df['NC'].std():.4f}")
    print(f"  True Positive Rate (TPR): {df['TPR'].mean():.4f} ± {df['TPR'].std():.4f}")
    print(f"  False Positive Rate (FPR): {df['FPR'].mean():.4f} ± {df['FPR'].std():.4f}")
    
    print(f"\n🔊 AUDIO QUALITY METRICS:")
    print(f"  Scale-invariant SNR (SiSNR): {df['SiSNR'].mean():.2f} ± {df['SiSNR'].std():.2f} dB")
    print(f"  Peak SNR (PSNR): {df['PSNR'].mean():.2f} ± {df['PSNR'].std():.2f} dB")
    print(f"  Perceptual Quality (PESQ): {df['PESQ'].mean():.3f} ± {df['PESQ'].std():.3f}")
    print(f"  Mean Squared Error (MSE): {df['MSE'].mean():.6f} ± {df['MSE'].std():.6f}")
    print(f"  Structural Similarity (SSIM): {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
    
    print(f"\n⚡ PROCESSING PERFORMANCE:")
    print(f"  Average embedding time: {df['embed_time'].mean():.3f} ± {df['embed_time'].std():.3f} seconds")
    print(f"  Average extraction time: {df['extract_time'].mean():.3f} ± {df['extract_time'].std():.3f} seconds")
    
    print(f"\n🏆 PERFORMANCE ASSESSMENT:")
    if df['BER'].mean() < 0.05:
        print(f"  ✓ BER: EXCELLENT (< 5% error rate)")
    elif df['BER'].mean() < 0.1:
        print(f"  ✓ BER: GOOD (< 10% error rate)")
    else:
        print(f"  ⚠️ BER: NEEDS IMPROVEMENT (≥ 10% error rate)")
    
    if df['NC'].mean() > 0.9:
        print(f"  ✓ NC: EXCELLENT (> 90% correlation)")
    elif df['NC'].mean() > 0.8:
        print(f"  ✓ NC: GOOD (> 80% correlation)")
    else:
        print(f"  ⚠️ NC: NEEDS IMPROVEMENT (≤ 80% correlation)")
    
    if df['PESQ'].mean() > 4.0:
        print(f"  ✓ PESQ: EXCELLENT (> 4.0 quality)")
    elif df['PESQ'].mean() > 3.0:
        print(f"  ✓ PESQ: GOOD (> 3.0 quality)")
    else:
        print(f"  ⚠️ PESQ: NEEDS IMPROVEMENT (≤ 3.0 quality)")


def save_summary_report(df: pd.DataFrame, output_dir: str):
    """Save a comprehensive summary report."""
    report_path = os.path.join(output_dir, "SVD_STFT_EVALUATION_REPORT.md")
    
    with open(report_path, 'w') as f:
        f.write("# SVD_STFT WATERMARKING METHOD EVALUATION REPORT\n\n")
        f.write("## Test Configuration\n")
        f.write(f"- **Method**: SVD_STFT\n")
        f.write(f"- **Total Files**: {len(df)}\n")
        f.write(f"- **Watermark Payload**: 64 bits per file\n")
        f.write(f"- **Sample Rate**: 16000 Hz\n")
        f.write(f"- **Calibration Target**: PESQ drop ≤ 0.05\n\n")
        
        f.write("## Key Performance Metrics\n\n")
        f.write("### Watermarking Reliability\n")
        f.write(f"- **Bit Error Rate (BER)**: {df['BER'].mean():.4f} ± {df['BER'].std():.4f}\n")
        f.write(f"- **Normalized Correlation (NC)**: {df['NC'].mean():.4f} ± {df['NC'].std():.4f}\n")
        f.write(f"- **True Positive Rate (TPR)**: {df['TPR'].mean():.4f} ± {df['TPR'].std():.4f}\n")
        f.write(f"- **False Positive Rate (FPR)**: {df['FPR'].mean():.4f} ± {df['FPR'].std():.4f}\n\n")
        
        f.write("### Audio Quality Metrics\n")
        f.write(f"- **Scale-invariant SNR (SiSNR)**: {df['SiSNR'].mean():.2f} ± {df['SiSNR'].std():.2f} dB\n")
        f.write(f"- **Peak SNR (PSNR)**: {df['PSNR'].mean():.2f} ± {df['PSNR'].std():.2f} dB\n")
        f.write(f"- **Perceptual Quality (PESQ)**: {df['PESQ'].mean():.3f} ± {df['PESQ'].std():.3f}\n")
        f.write(f"- **Mean Squared Error (MSE)**: {df['MSE'].mean():.6f} ± {df['MSE'].std():.6f}\n")
        f.write(f"- **Structural Similarity (SSIM)**: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}\n\n")
        
        f.write("### Processing Performance\n")
        f.write(f"- **Average Embedding Time**: {df['embed_time'].mean():.3f} ± {df['embed_time'].std():.3f} seconds\n")
        f.write(f"- **Average Extraction Time**: {df['extract_time'].mean():.3f} ± {df['extract_time'].std():.3f} seconds\n\n")
        
        f.write("## Conclusion\n")
        f.write("This report provides comprehensive evaluation metrics for the SVD_STFT watermarking method.\n")
        f.write("All metrics indicate the method's performance across different audio types and conditions.\n")
    
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducibility
    test_svdstft_only()

#!/usr/bin/env python3
"""
Comprehensive testing of SVD_STFT watermarking method against other methods.
Tests on 100 sample audio files with fair calibration and detailed metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.utils.watermark_methods import get_all_methods, WatermarkMethod
from src.utils.metrics import evaluate_performance, compute_pesq, format_metrics_for_csv
from src.audioio.audio_io import load_audio, save_audio, preprocess


class ComprehensiveWatermarkTester:
    """Comprehensive testing framework for watermarking methods."""
    
    def __init__(self, audio_dir: str, output_dir: str, sample_rate: int = 16000):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.methods = get_all_methods()
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "watermarked").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Results storage
        self.results = []
        self.calibration_results = []
        
    def get_audio_files(self) -> List[Path]:
        """Get all audio files from the directory."""
        audio_files = []
        
        # Get speech files
        speech_dir = self.audio_dir / "speech_wav"
        if speech_dir.exists():
            audio_files.extend(list(speech_dir.glob("*.wav")))
            
        # Get music files
        music_dir = self.audio_dir / "music_wav"
        if music_dir.exists():
            audio_files.extend(list(music_dir.glob("*.wav")))
            
        return sorted(audio_files)
    
    def generate_watermark_bits(self, length: int = 64) -> List[int]:
        """Generate random watermark bits."""
        random.seed(42)  # Fixed seed for reproducibility
        return [random.randint(0, 1) for _ in range(length)]
    
    def calibrate_method(self, method: WatermarkMethod, audio: np.ndarray, 
                        target_pesq_drop: float = 0.05) -> Dict[str, Any]:
        """Calibrate a watermarking method to achieve target PESQ drop."""
        print(f"Calibrating {method.name}...")
        
        start_time = time.time()
        
        try:
            # Use first 5 seconds for calibration
            test_audio = audio[:min(len(audio), 5 * self.sample_rate)]
            
            # Generate test watermark
            test_bits = self.generate_watermark_bits(32)
            
            # Calibrate method
            alpha = method.calibrate(test_audio, self.sample_rate, target_pesq_drop)
            
            # Verify calibration
            watermarked, metadata = method.embed(test_audio, self.sample_rate, test_bits)
            
            # Compute PESQ drop
            pesq_orig = compute_pesq(test_audio, test_audio, self.sample_rate)
            pesq_wm = compute_pesq(test_audio, watermarked, self.sample_rate)
            pesq_drop = pesq_orig - pesq_wm
            
            # Extract watermark to verify functionality
            extracted = method.extract(watermarked, self.sample_rate, len(test_bits), metadata)
            ber = sum(b1 != b2 for b1, b2 in zip(test_bits, extracted)) / len(test_bits)
            
            calibration_time = time.time() - start_time
            
            result = {
                'method': method.name,
                'alpha': alpha,
                'target_pesq_drop': target_pesq_drop,
                'actual_pesq_drop': pesq_drop,
                'calibration_ber': ber,
                'calibration_time': calibration_time,
                'success': True
            }
            
            print(f"  ✓ {method.name} calibrated: alpha={alpha:.6f}, PESQ drop={pesq_drop:.4f}, BER={ber:.4f}")
            
        except Exception as e:
            calibration_time = time.time() - start_time
            result = {
                'method': method.name,
                'alpha': None,
                'target_pesq_drop': target_pesq_drop,
                'actual_pesq_drop': None,
                'calibration_ber': None,
                'calibration_time': calibration_time,
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ {method.name} calibration failed: {e}")
        
        return result
    
    def test_single_file(self, audio_file: Path, method: WatermarkMethod, 
                         watermark_bits: List[int]) -> Dict[str, Any]:
        """Test a single audio file with a watermarking method."""
        try:
            # Load and preprocess audio
            audio = load_audio(str(audio_file), self.sample_rate)
            audio = preprocess(audio)
            
            # Embed watermark
            start_time = time.time()
            watermarked, metadata = method.embed(audio, self.sample_rate, watermark_bits)
            embed_time = time.time() - start_time
            
            # Extract watermark
            start_time = time.time()
            extracted = method.extract(watermarked, self.sample_rate, len(watermark_bits), metadata)
            extract_time = time.time() - start_time
            
            # Compute metrics
            metrics = evaluate_performance(audio, watermarked, extracted, watermark_bits, self.sample_rate)
            
            # Save watermarked audio
            output_filename = f"{audio_file.stem}_{method.name}_watermarked.wav"
            output_path = self.output_dir / "watermarked" / output_filename
            save_audio(watermarked, self.sample_rate, str(output_path))
            
            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump({
                    'method': method.name,
                    'alpha': metadata.get('alpha'),
                    'original_file': str(audio_file),
                    'watermark_bits': watermark_bits,
                    'extracted_bits': extracted,
                    'metrics': metrics
                }, f, indent=2)
            
            result = {
                'file': audio_file.name,
                'method': method.name,
                'file_size_mb': os.path.getsize(audio_file) / (1024 * 1024),
                'duration_seconds': len(audio) / self.sample_rate,
                'embed_time': embed_time,
                'extract_time': extract_time,
                **metrics
            }
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name} with {method.name}: {e}")
            return {
                'file': audio_file.name,
                'method': method.name,
                'error': str(e)
            }
    
    def run_comprehensive_test(self, max_files: int = None, target_pesq_drop: float = 0.05):
        """Run comprehensive testing on all methods and audio files."""
        print("=" * 80)
        print("COMPREHENSIVE WATERMARKING METHOD EVALUATION")
        print("=" * 80)
        
        # Get audio files
        audio_files = self.get_audio_files()
        if max_files:
            audio_files = audio_files[:max_files]
        
        print(f"Found {len(audio_files)} audio files for testing")
        print(f"Testing {len(self.methods)} watermarking methods")
        print(f"Target PESQ drop: {target_pesq_drop}")
        print()
        
        # Step 1: Calibrate all methods
        print("STEP 1: CALIBRATING METHODS")
        print("-" * 40)
        
        for method in self.methods:
            # Use first audio file for calibration
            if audio_files:
                cal_audio = load_audio(str(audio_files[0]), self.sample_rate)
                cal_audio = preprocess(cal_audio)
                
                cal_result = self.calibrate_method(method, cal_audio, target_pesq_drop)
                self.calibration_results.append(cal_result)
                
                if not cal_result['success']:
                    print(f"Warning: {method.name} failed calibration, skipping...")
                    self.methods.remove(method)
        
        print(f"\nSuccessfully calibrated {len(self.methods)} methods")
        print()
        
        # Step 2: Test all methods on all files
        print("STEP 2: TESTING METHODS ON AUDIO FILES")
        print("-" * 40)
        
        watermark_bits = self.generate_watermark_bits(64)
        print(f"Using watermark: {''.join(map(str, watermark_bits[:16]))}... (64 bits total)")
        print()
        
        total_tests = len(audio_files) * len(self.methods)
        current_test = 0
        
        for i, audio_file in enumerate(audio_files):
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
            
            for method in self.methods:
                current_test += 1
                print(f"  [{current_test}/{total_tests}] Testing {method.name}...", end=" ")
                
                result = self.test_single_file(audio_file, method, watermark_bits)
                
                if 'error' not in result:
                    print("✓")
                    self.results.append(result)
                else:
                    print("✗")
        
        print(f"\nCompleted {len(self.results)} successful tests")
        print()
        
        # Step 3: Generate results
        print("STEP 3: GENERATING RESULTS")
        print("-" * 40)
        
        self.generate_results()
        
        print("Testing completed successfully!")
        print(f"Results saved to: {self.output_dir}")
    
    def generate_results(self):
        """Generate comprehensive results and save to files."""
        if not self.results:
            print("No results to generate")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        results_file = self.output_dir / "results" / "comprehensive_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")
        
        # Generate summary statistics
        summary = self.generate_summary_statistics(df)
        summary_file = self.output_dir / "results" / "summary_statistics.csv"
        summary.to_csv(summary_file, index=True)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Generate method comparison
        comparison = self.generate_method_comparison(df)
        comparison_file = self.output_dir / "results" / "method_comparison.csv"
        comparison.to_csv(comparison_file, index=False)
        print(f"Method comparison saved to: {comparison_file}")
        
        # Save calibration results
        cal_df = pd.DataFrame(self.calibration_results)
        cal_file = self.output_dir / "results" / "calibration_results.csv"
        cal_df.to_csv(cal_file, index=False)
        print(f"Calibration results saved to: {cal_file}")
        
        # Print summary
        self.print_summary(df)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for each method."""
        summary_stats = []
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            stats = {
                'Method': method,
                'Files_Processed': len(method_data),
                'Avg_SiSNR_dB': method_data['SiSNR'].mean(),
                'Std_SiSNR_dB': method_data['SiSNR'].std(),
                'Avg_BER': method_data['BER'].mean(),
                'Std_BER': method_data['BER'].std(),
                'Avg_NC': method_data['NC'].mean(),
                'Std_NC': method_data['NC'].std(),
                'Avg_TPR': method_data['TPR'].mean(),
                'Std_TPR': method_data['TPR'].std(),
                'Avg_FPR': method_data['FPR'].mean(),
                'Std_FPR': method_data['FPR'].std(),
                'Avg_PESQ': method_data['PESQ'].mean(),
                'Std_PESQ': method_data['PESQ'].std(),
                'Avg_PSNR_dB': method_data['PSNR'].mean(),
                'Std_PSNR_dB': method_data['PSNR'].std(),
                'Avg_Embed_Time_s': method_data['embed_time'].mean(),
                'Avg_Extract_Time_s': method_data['extract_time'].mean()
            }
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def generate_method_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate method comparison table."""
        comparison = []
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            # Best performance in each category
            best_sisnr = method_data.loc[method_data['SiSNR'].idxmax()]
            best_ber = method_data.loc[method_data['BER'].idxmin()]
            best_nc = method_data.loc[method_data['NC'].idxmax()]
            best_tpr = method_data.loc[method_data['TPR'].idxmax()]
            best_fpr = method_data.loc[method_data['FPR'].idxmin()]
            best_pesq = method_data.loc[method_data['PESQ'].idxmax()]
            
            comp = {
                'Method': method,
                'Best_SiSNR_File': best_sisnr['file'],
                'Best_SiSNR_dB': best_sisnr['SiSNR'],
                'Best_BER_File': best_ber['file'],
                'Best_BER': best_ber['BER'],
                'Best_NC_File': best_nc['file'],
                'Best_NC': best_nc['NC'],
                'Best_TPR_File': best_tpr['file'],
                'Best_TPR': best_tpr['TPR'],
                'Best_FPR_File': best_fpr['file'],
                'Best_FPR': best_fpr['FPR'],
                'Best_PESQ_File': best_pesq['file'],
                'Best_PESQ': best_pesq['PESQ'],
                'Total_Files': len(method_data)
            }
            
            comparison.append(comp)
        
        return pd.DataFrame(comparison)
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            print(f"\n{method}:")
            print(f"  Files processed: {len(method_data)}")
            print(f"  Average SiSNR: {method_data['SiSNR'].mean():.2f} dB")
            print(f"  Average BER: {method_data['BER'].mean():.6f}")
            print(f"  Average NC: {method_data['NC'].mean():.4f}")
            print(f"  Average TPR: {method_data['TPR'].mean():.4f}")
            print(f"  Average FPR: {method_data['FPR'].mean():.4f}")
            print(f"  Average PESQ: {method_data['PESQ'].mean():.3f}")
            print(f"  Average PSNR: {method_data['PSNR'].mean():.2f} dB")


def main():
    """Main function to run the comprehensive test."""
    # Configuration
    audio_dir = "100sample_wav"
    output_dir = "comprehensive_test_results_updated"
    sample_rate = 16000
    max_files = 100  # Set to None to test all files
    target_pesq_drop = 0.05
    
    # Check if audio directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        return
    
    # Create tester and run
    tester = ComprehensiveWatermarkTester(audio_dir, output_dir, sample_rate)
    
    try:
        tester.run_comprehensive_test(max_files, target_pesq_drop)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

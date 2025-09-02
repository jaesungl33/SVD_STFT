#!/usr/bin/env python3
"""
Comprehensive robustness evaluation script for SVD_STFT watermarking.
Tests watermark robustness against various audio attacks similar to AudioSeal.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import embed_svd_stft, extract_svd_stft, compute_stft
from stft.stft_transform import reconstruct_audio
from audioio.audio_io import load_audio, save_audio


class SVDSTFTRobustnessTester:
    """Comprehensive robustness tester for SVD_STFT watermarking."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.evaluator = RobustnessEvaluator(sample_rate)
        
    def create_watermark_bits(self, num_bits: int = 64) -> List[int]:
        """Create a random watermark bit pattern."""
        return np.random.randint(0, 2, num_bits).tolist()
    
    def embed_watermark(self, audio: np.ndarray, watermark_bits: List[int], 
                       alpha: float = 0.1, block_size: Tuple[int, int] = (8, 8),
                       key: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed watermark using SVD_STFT method."""
        # Compute STFT
        S = compute_stft(audio, self.sample_rate, n_fft=256, hop_length=64, window='hann')
        
        # Embed watermark
        S_watermarked, sigma_ref = embed_svd_stft(S, watermark_bits, alpha, block_size, key)
        
        # Reconstruct audio
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        
        # Store parameters for extraction
        extract_params = {
            'alpha': alpha,
            'block_size': block_size,
            'key': key,
            'num_bits': len(watermark_bits),
            'sigma_ref': sigma_ref
        }
        
        return watermarked_audio, extract_params
    
    def extract_watermark(self, audio: np.ndarray, extract_params: Dict[str, Any]) -> List[int]:
        """Extract watermark from audio."""
        # Compute STFT
        S = compute_stft(audio, self.sample_rate, n_fft=256, hop_length=64, window='hann')
        
        # Extract watermark
        extracted_bits = extract_svd_stft(
            S, 
            extract_params['alpha'],
            extract_params['block_size'],
            extract_params['key'],
            extract_params['num_bits'],
            sigma_ref=extract_params['sigma_ref']
        )
        
        return extracted_bits
    
    def test_single_file(self, audio_path: str, watermark_bits: List[int],
                        alpha: float = 0.1, block_size: Tuple[int, int] = (8, 8),
                        key: int = 42, test_config: str = "evaluation") -> Dict[str, Any]:
        """Test robustness on a single audio file."""
        print(f"Testing file: {audio_path}")
        
        # Load audio
        audio = load_audio(audio_path, self.sample_rate)
        
        # Embed watermark
        watermarked_audio, extract_params = self.embed_watermark(
            audio, watermark_bits, alpha, block_size, key
        )
        
        # Test extraction on clean watermarked audio
        clean_extracted = self.extract_watermark(watermarked_audio, extract_params)
        clean_ber = self.evaluator._calculate_ber(watermark_bits, clean_extracted)
        
        # Evaluate robustness
        results = self.evaluator.evaluate_robustness(
            original_audio=audio,
            watermarked_audio=watermarked_audio,
            extract_function=self.extract_watermark,
            extract_params=extract_params,
            watermark_bits=watermark_bits,
            test_config=test_config
        )
        
        # Add clean extraction result
        results['clean'] = {
            'ber': clean_ber,
            'snr': self.evaluator._calculate_snr(audio, watermarked_audio),
            'success': clean_ber < 0.1
        }
        
        return {
            'file_path': audio_path,
            'results': results,
            'extract_params': extract_params,
            'watermark_bits': watermark_bits
        }
    
    def test_multiple_files(self, audio_files: List[str], watermark_bits: List[int],
                           alpha: float = 0.1, block_size: Tuple[int, int] = (8, 8),
                           key: int = 42, test_config: str = "evaluation") -> List[Dict[str, Any]]:
        """Test robustness on multiple audio files."""
        all_results = []
        
        for audio_file in tqdm(audio_files, desc="Testing files"):
            try:
                result = self.test_single_file(
                    audio_file, watermark_bits, alpha, block_size, key, test_config
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error testing {audio_file}: {e}")
                continue
        
        return all_results
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple files."""
        if not results:
            return {}
        
        # Get all attack types
        attack_types = list(results[0]['results'].keys())
        
        aggregated = {}
        
        for attack_type in attack_types:
            bers = []
            snrs = []
            successes = []
            
            for result in results:
                if attack_type in result['results']:
                    attack_result = result['results'][attack_type]
                    bers.append(attack_result.get('ber', 1.0))
                    snrs.append(attack_result.get('snr', -np.inf))
                    successes.append(attack_result.get('success', False))
            
            if bers:
                aggregated[attack_type] = {
                    'mean_ber': np.mean(bers),
                    'std_ber': np.std(bers),
                    'median_ber': np.median(bers),
                    'mean_snr': np.mean(snrs),
                    'std_snr': np.std(snrs),
                    'success_rate': np.mean(successes) * 100,
                    'num_tests': len(bers)
                }
        
        return aggregated
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_file = output_path / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create aggregated results
        aggregated = self.aggregate_results(results)
        
        # Save aggregated results
        aggregated_file = output_path / "aggregated_results.json"
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        # Create CSV summary
        summary_data = []
        for attack_type, stats in aggregated.items():
            summary_data.append({
                'attack': attack_type,
                'mean_ber': stats['mean_ber'],
                'std_ber': stats['std_ber'],
                'median_ber': stats['median_ber'],
                'mean_snr': stats['mean_snr'],
                'std_snr': stats['std_snr'],
                'success_rate': stats['success_rate'],
                'num_tests': stats['num_tests']
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = output_path / "robustness_summary.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate and save report
        report = self.generate_comprehensive_report(aggregated, len(results))
        report_file = output_path / "robustness_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Create visualization
        self.create_visualization(aggregated, output_path)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def generate_comprehensive_report(self, aggregated: Dict[str, Any], num_files: int) -> str:
        """Generate a comprehensive robustness report."""
        report = "=" * 80 + "\n"
        report += "SVD_STFT WATERMARKING ROBUSTNESS EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Number of test files: {num_files}\n"
        report += f"Sample rate: {self.sample_rate} Hz\n\n"
        
        # Overall statistics
        success_rates = [stats['success_rate'] for stats in aggregated.values()]
        overall_success_rate = np.mean(success_rates)
        report += f"Overall Success Rate: {overall_success_rate:.1f}%\n\n"
        
        # Detailed results table
        report += "DETAILED RESULTS:\n"
        report += "-" * 100 + "\n"
        report += f"{'Attack':<20} {'Mean BER':<12} {'Std BER':<12} {'Success Rate':<15} {'Mean SNR':<12}\n"
        report += "-" * 100 + "\n"
        
        for attack_type, stats in aggregated.items():
            report += f"{attack_type:<20} {stats['mean_ber']:<12.4f} {stats['std_ber']:<12.4f} "
            report += f"{stats['success_rate']:<15.1f} {stats['mean_snr']:<12.2f}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def create_visualization(self, aggregated: Dict[str, Any], output_path: Path):
        """Create visualization of results."""
        # Prepare data for plotting
        attacks = list(aggregated.keys())
        mean_bers = [aggregated[attack]['mean_ber'] for attack in attacks]
        success_rates = [aggregated[attack]['success_rate'] for attack in attacks]
        mean_snrs = [aggregated[attack]['mean_snr'] for attack in attacks]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Mean BER by attack
        ax1.bar(attacks, mean_bers, color='skyblue', alpha=0.7)
        ax1.set_title('Mean Bit Error Rate by Attack')
        ax1.set_ylabel('Mean BER')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate by attack
        ax2.bar(attacks, success_rates, color='lightgreen', alpha=0.7)
        ax2.set_title('Success Rate by Attack')
        ax2.set_ylabel('Success Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean SNR by attack
        ax3.bar(attacks, mean_snrs, color='salmon', alpha=0.7)
        ax3.set_title('Mean SNR by Attack')
        ax3.set_ylabel('Mean SNR (dB)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: BER vs Success Rate scatter
        ax4.scatter(mean_bers, success_rates, s=100, alpha=0.7, c='purple')
        ax4.set_xlabel('Mean BER')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('BER vs Success Rate')
        ax4.grid(True, alpha=0.3)
        
        # Add attack labels to scatter plot
        for i, attack in enumerate(attacks):
            ax4.annotate(attack, (mean_bers[i], success_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / "robustness_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()


def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all audio files in a directory."""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(Path(directory).rglob(f"*{ext}"))
    
    return [str(f) for f in audio_files]


def main():
    parser = argparse.ArgumentParser(description="SVD_STFT Robustness Evaluation")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing audio files to test")
    parser.add_argument("--output_dir", type=str, default="robustness_results",
                       help="Output directory for results")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Target sample rate")
    parser.add_argument("--num_bits", type=int, default=64,
                       help="Number of watermark bits")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Watermark strength parameter")
    parser.add_argument("--block_size", type=str, default="8,8",
                       help="Block size as 'rows,cols'")
    parser.add_argument("--key", type=int, default=42,
                       help="Random seed for watermark generation")
    parser.add_argument("--test_config", type=str, default="evaluation",
                       choices=["training", "evaluation"],
                       help="Test configuration (training or evaluation)")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to test")
    
    args = parser.parse_args()
    
    # Parse block size
    block_size = tuple(map(int, args.block_size.split(',')))
    
    # Find audio files
    print(f"Searching for audio files in: {args.input_dir}")
    audio_files = find_audio_files(args.input_dir)
    
    if not audio_files:
        print("No audio files found!")
        return
    
    if args.max_files:
        audio_files = audio_files[:args.max_files]
    
    print(f"Found {len(audio_files)} audio files")
    
    # Initialize tester
    tester = SVDSTFTRobustnessTester(sample_rate=args.sample_rate)
    
    # Create watermark bits
    watermark_bits = tester.create_watermark_bits(args.num_bits)
    print(f"Generated watermark with {len(watermark_bits)} bits")
    
    # Run tests
    print(f"Starting robustness evaluation with {args.test_config} configuration...")
    results = tester.test_multiple_files(
        audio_files, watermark_bits, args.alpha, block_size, 
        args.key, args.test_config
    )
    
    if not results:
        print("No successful tests completed!")
        return
    
    # Save results
    output_path = tester.save_results(results, args.output_dir)
    
    # Print summary
    aggregated = tester.aggregate_results(results)
    success_rates = [stats['success_rate'] for stats in aggregated.values()]
    overall_success_rate = np.mean(success_rates)
    
    print(f"\nEvaluation completed!")
    print(f"Overall success rate: {overall_success_rate:.1f}%")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

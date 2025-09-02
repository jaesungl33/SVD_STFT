#!/usr/bin/env python3
"""
Parameter optimization script for SVD_STFT watermarking.
Systematically tests different parameter combinations to find optimal settings for robustness.
"""

import sys
import numpy as np
import soundfile as sf
import json
import pandas as pd
from pathlib import Path
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import embed_svd_stft, extract_svd_stft, compute_stft
from stft.stft_transform import reconstruct_audio


def create_test_audio(duration: float = 10.0, sample_rate: int = 16000) -> np.ndarray:
    """Create a complex test audio signal with multiple frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex signal with multiple frequencies
    audio = (0.4 * np.sin(2 * np.pi * 440 * t) +      # A4 note
             0.3 * np.sin(2 * np.pi * 880 * t) +      # A5 note
             0.2 * np.sin(2 * np.pi * 1760 * t) +     # A6 note
             0.1 * np.sin(2 * np.pi * 220 * t) +      # A3 note
             0.05 * np.sin(2 * np.pi * 110 * t))      # A2 note
    
    # Add some amplitude modulation
    audio *= (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
    
    return audio


def test_parameter_combination(audio: np.ndarray, watermark_bits: list, 
                             alpha: float, block_size: tuple, key: int,
                             sample_rate: int = 16000) -> dict:
    """Test a single parameter combination."""
    try:
        # Embed watermark
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, sigma_ref = embed_svd_stft(S, watermark_bits, alpha, block_size, key)
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        audio = audio[:min_len]
        
        # Test clean extraction
        S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        extracted_bits = extract_svd_stft(
            S_test, 
            alpha,
            block_size,
            key,
            len(watermark_bits),
            sigma_ref=sigma_ref
        )
        
        # Calculate clean BER
        clean_ber = sum(1 for i in range(len(watermark_bits)) 
                       if watermark_bits[i] != extracted_bits[i]) / len(watermark_bits)
        
        # Calculate SNR
        noise = audio - watermarked_audio
        signal_power = np.sum(audio ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'alpha': alpha,
            'block_size': block_size,
            'clean_ber': clean_ber,
            'snr': snr,
            'success': clean_ber < 0.1,
            'sigma_ref': sigma_ref,
            'extract_params': {
                'alpha': alpha,
                'block_size': block_size,
                'key': key,
                'num_bits': len(watermark_bits),
                'sigma_ref': sigma_ref
            }
        }
        
    except Exception as e:
        return {
            'alpha': alpha,
            'block_size': block_size,
            'clean_ber': 1.0,
            'snr': -np.inf,
            'success': False,
            'error': str(e)
        }


def optimize_parameters():
    """Main parameter optimization function."""
    print("=" * 80)
    print("SVD_STFT PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Create test audio
    sample_rate = 16000
    audio = create_test_audio(duration=8.0, sample_rate=sample_rate)
    print(f"Created test audio: {len(audio)/sample_rate:.1f} seconds")
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 32).tolist()
    print(f"Generated watermark: {len(watermark_bits)} bits")
    
    # Define parameter ranges to test
    alpha_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    block_sizes = [(4, 4), (8, 8), (12, 12), (16, 16), (20, 20), (24, 24)]
    keys = [42]  # We'll use a fixed key for consistency
    
    print(f"\nTesting {len(alpha_values)} alpha values and {len(block_sizes)} block sizes")
    print(f"Total combinations: {len(alpha_values) * len(block_sizes)}")
    
    # Test all parameter combinations
    results = []
    total_combinations = len(alpha_values) * len(block_sizes)
    
    for alpha, block_size in tqdm(product(alpha_values, block_sizes), 
                                 total=total_combinations, 
                                 desc="Testing parameters"):
        result = test_parameter_combination(
            audio, watermark_bits, alpha, block_size, keys[0], sample_rate
        )
        results.append(result)
    
    # Filter successful combinations
    successful_results = [r for r in results if r.get('success', False)]
    print(f"\nSuccessful combinations: {len(successful_results)}/{len(results)}")
    
    if not successful_results:
        print("No successful combinations found. Showing best results...")
        # Sort by clean BER (lower is better)
        results.sort(key=lambda x: x.get('clean_ber', 1.0))
        best_results = results[:10]
    else:
        # Sort successful results by SNR (higher is better)
        successful_results.sort(key=lambda x: x.get('snr', -np.inf), reverse=True)
        best_results = successful_results[:10]
    
    # Display best results
    print("\n" + "=" * 80)
    print("BEST PARAMETER COMBINATIONS")
    print("=" * 80)
    print(f"{'Alpha':<8} {'Block Size':<12} {'Clean BER':<12} {'SNR (dB)':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in best_results:
        alpha = result['alpha']
        block_size = result['block_size']
        clean_ber = result.get('clean_ber', 1.0)
        snr = result.get('snr', -np.inf)
        success = "✓" if result.get('success', False) else "✗"
        
        print(f"{alpha:<8.2f} {str(block_size):<12} {clean_ber:<12.4f} {snr:<12.2f} {success:<10}")
    
    # Test robustness for best combinations
    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTING FOR BEST COMBINATIONS")
    print("=" * 80)
    
    evaluator = RobustnessEvaluator(sample_rate)
    
    def extract_function(audio_signal, **params):
        """Extract watermark from audio."""
        S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
        return extract_svd_stft(
            S, 
            params['alpha'],
            params['block_size'],
            params['key'],
            params['num_bits'],
            sigma_ref=params['sigma_ref']
        )
    
    robustness_results = []
    
    for i, result in enumerate(best_results[:5]):  # Test top 5 combinations
        print(f"\nTesting combination {i+1}: α={result['alpha']}, block={result['block_size']}")
        
        # Re-embed with these parameters
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, sigma_ref = embed_svd_stft(
            S, watermark_bits, 
            result['alpha'], 
            result['block_size'], 
            keys[0]
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        
        # Test robustness
        attack_results = evaluator.evaluate_robustness(
            original_audio=audio,
            watermarked_audio=watermarked_audio,
            extract_function=extract_function,
            extract_params=result['extract_params'],
            watermark_bits=watermark_bits,
            test_config="evaluation"
        )
        
        # Calculate overall success rate
        successful_attacks = sum(1 for r in attack_results.values() if r.get('success', False))
        total_attacks = len(attack_results)
        success_rate = (successful_attacks / total_attacks) * 100
        
        # Calculate average BER across attacks
        avg_ber = np.mean([r.get('ber', 1.0) for r in attack_results.values()])
        
        robustness_results.append({
            'combination': i+1,
            'alpha': result['alpha'],
            'block_size': result['block_size'],
            'clean_ber': result.get('clean_ber', 1.0),
            'clean_snr': result.get('snr', -np.inf),
            'attack_success_rate': success_rate,
            'avg_attack_ber': avg_ber,
            'attack_results': attack_results
        })
        
        print(f"  Clean BER: {result.get('clean_ber', 1.0):.4f}")
        print(f"  Attack Success Rate: {success_rate:.1f}%")
        print(f"  Average Attack BER: {avg_ber:.4f}")
    
    # Find best overall combination
    if robustness_results:
        # Sort by attack success rate, then by clean BER
        robustness_results.sort(key=lambda x: (x['attack_success_rate'], -x['clean_ber']), reverse=True)
        best_overall = robustness_results[0]
        
        print("\n" + "=" * 80)
        print("BEST OVERALL COMBINATION")
        print("=" * 80)
        print(f"Alpha: {best_overall['alpha']}")
        print(f"Block Size: {best_overall['block_size']}")
        print(f"Clean BER: {best_overall['clean_ber']:.4f}")
        print(f"Clean SNR: {best_overall['clean_snr']:.2f} dB")
        print(f"Attack Success Rate: {best_overall['attack_success_rate']:.1f}%")
        print(f"Average Attack BER: {best_overall['avg_attack_ber']:.4f}")
    
    # Save results
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "parameter_optimization_results.json", 'w') as f:
        json.dump({
            'all_results': results,
            'best_results': best_results,
            'robustness_results': robustness_results,
            'best_overall': best_overall if robustness_results else None
        }, f, indent=2, default=str)
    
    # Create CSV summary
    summary_data = []
    for result in robustness_results:
        summary_data.append({
            'combination': result['combination'],
            'alpha': result['alpha'],
            'block_size': str(result['block_size']),
            'clean_ber': result['clean_ber'],
            'clean_snr': result['clean_snr'],
            'attack_success_rate': result['attack_success_rate'],
            'avg_attack_ber': result['avg_attack_ber']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "optimization_summary.csv", index=False)
    
    # Create visualization
    create_optimization_visualization(results, robustness_results, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    
    return best_overall if robustness_results else None


def create_optimization_visualization(all_results, robustness_results, output_dir):
    """Create visualizations for optimization results."""
    # Prepare data
    alphas = [r['alpha'] for r in all_results]
    block_sizes = [str(r['block_size']) for r in all_results]
    clean_bers = [r.get('clean_ber', 1.0) for r in all_results]
    snrs = [r.get('snr', -np.inf) for r in all_results]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Clean BER vs Alpha
    unique_alphas = sorted(set(alphas))
    ber_by_alpha = []
    for alpha in unique_alphas:
        alpha_results = [r for r in all_results if r['alpha'] == alpha]
        avg_ber = np.mean([r.get('clean_ber', 1.0) for r in alpha_results])
        ber_by_alpha.append(avg_ber)
    
    ax1.plot(unique_alphas, ber_by_alpha, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Average Clean BER')
    ax1.set_title('Clean BER vs Alpha')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SNR vs Alpha
    snr_by_alpha = []
    for alpha in unique_alphas:
        alpha_results = [r for r in all_results if r['alpha'] == alpha]
        avg_snr = np.mean([r.get('snr', -np.inf) for r in alpha_results])
        snr_by_alpha.append(avg_snr)
    
    ax2.plot(unique_alphas, snr_by_alpha, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Average SNR (dB)')
    ax2.set_title('SNR vs Alpha')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Attack Success Rate vs Clean BER
    if robustness_results:
        attack_rates = [r['attack_success_rate'] for r in robustness_results]
        clean_bers = [r['clean_ber'] for r in robustness_results]
        
        ax3.scatter(clean_bers, attack_rates, s=100, alpha=0.7, c='green')
        ax3.set_xlabel('Clean BER')
        ax3.set_ylabel('Attack Success Rate (%)')
        ax3.set_title('Attack Success Rate vs Clean BER')
        ax3.grid(True, alpha=0.3)
        
        # Add labels
        for i, result in enumerate(robustness_results):
            ax3.annotate(f"α={result['alpha']}", 
                        (clean_bers[i], attack_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Parameter space heatmap
    if robustness_results:
        # Create pivot table for heatmap
        pivot_data = []
        for result in robustness_results:
            pivot_data.append({
                'alpha': result['alpha'],
                'block_size': str(result['block_size']),
                'success_rate': result['attack_success_rate']
            })
        
        df_pivot = pd.DataFrame(pivot_data)
        pivot_table = df_pivot.pivot(index='alpha', columns='block_size', values='success_rate')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Attack Success Rate Heatmap')
        ax4.set_xlabel('Block Size')
        ax4.set_ylabel('Alpha')
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    best_params = optimize_parameters()
    
    if best_params:
        print(f"\n🎯 RECOMMENDED PARAMETERS:")
        print(f"   Alpha: {best_params['alpha']}")
        print(f"   Block Size: {best_params['block_size']}")
        print(f"   Expected Clean BER: {best_params['clean_ber']:.4f}")
        print(f"   Expected Attack Success Rate: {best_params['attack_success_rate']:.1f}%")
    else:
        print("\n❌ No optimal parameters found. Consider expanding parameter ranges.")

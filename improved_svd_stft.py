#!/usr/bin/env python3
"""
Improved SVD_STFT watermarking implementation with better parameter tuning.
Addresses extraction issues and provides more robust watermarking.
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


def improved_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float, 
                           block_size: tuple, key: int, midband_ratio: tuple = None) -> tuple:
    """
    Improved SVD_STFT embedding with better parameter handling.
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    if n_blocks < len(bits):
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(bits)} bits")
    
    # Filter blocks by midband if specified
    if midband_ratio:
        rows, cols = S_mag.shape
        block_rows, block_cols = block_size
        fmin = int(rows * max(0.0, min(1.0, midband_ratio[0])))
        fmax = int(rows * max(0.0, min(1.0, midband_ratio[1])))
        allowed_block_rows = set()
        r = 0
        while r + block_rows <= rows:
            row_center = r + block_rows // 2
            if fmin <= row_center < fmax:
                allowed_block_rows.add(r)
            r += block_rows
        
        # Filter blocks
        filtered_blocks = []
        filtered_indices = []
        r = 0
        idx = 0
        while r + block_rows <= rows:
            c = 0
            row_allowed = r in allowed_block_rows
            while c + block_cols <= cols:
                if row_allowed:
                    filtered_blocks.append(blocks[idx])
                    filtered_indices.append(idx)
                idx += 1
                c += block_cols
            r += block_rows
        
        blocks = filtered_blocks
        n_blocks = len(blocks)
        
        if n_blocks < len(bits):
            raise ValueError(f"Not enough midband-filtered blocks ({n_blocks}) for {len(bits)} bits")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store original singular values for reference
    sigma_ref = []
    
    # Embed watermark
    for i, bit in enumerate(bits):
        if i >= len(perm):
            break
        
        block_idx = perm[i]
        block = blocks[block_idx]
        
        # Compute SVD
        U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
        
        # Store reference singular value
        sigma_ref.append(float(Sigma[0]))
        
        # Modify singular value based on bit
        if bit == 1:
            Sigma[0] *= (1 + alpha)
        else:
            Sigma[0] *= (1 - alpha)
        
        # Reconstruct block
        modified_block = U @ np.diag(Sigma) @ Vt
        blocks[block_idx] = modified_block
    
    # Reassemble
    shape = S_mag.shape
    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    return S_complex_mod, sigma_ref


def improved_extract_svd_stft(S_complex_mod: np.ndarray, alpha: float, 
                             block_size: tuple, key: int, num_bits: int,
                             sigma_ref: list = None, midband_ratio: tuple = None) -> list:
    """
    Improved SVD_STFT extraction with better threshold handling.
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    rows, cols = S_mag_mod.shape
    block_rows, block_cols = block_size
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    
    # Filter blocks by midband if specified
    if midband_ratio:
        fmin = int(rows * max(0.0, min(1.0, midband_ratio[0])))
        fmax = int(rows * max(0.0, min(1.0, midband_ratio[1])))
        allowed_block_rows = set()
        r = 0
        while r + block_rows <= rows:
            row_center = r + block_rows // 2
            if fmin <= row_center < fmax:
                allowed_block_rows.add(r)
            r += block_rows
        
        # Filter blocks
        filtered_blocks = []
        filtered_indices = []
        r = 0
        idx = 0
        while r + block_rows <= rows:
            c = 0
            row_allowed = r in allowed_block_rows
            while c + block_cols <= cols:
                if row_allowed:
                    filtered_blocks.append(blocks_mod[idx])
                    filtered_indices.append(idx)
                idx += 1
                c += block_cols
            r += block_rows
        
        blocks_mod = filtered_blocks
        n_blocks = len(blocks_mod)
        
        if n_blocks < num_bits:
            raise ValueError(f"Not enough midband-filtered blocks ({n_blocks}) for {num_bits} bits")
    
    # Generate permutation
    perm = pseudo_permutation(len(blocks_mod), key)
    
    # Extract bits
    extracted_bits = []
    sigmas = []
    
    for i in range(num_bits):
        if i >= len(perm):
            break
        
        block_idx = perm[i]
        block = blocks_mod[block_idx]
        
        # Compute SVD
        U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
        sigmas.append(float(Sigma[0]))
    
    # Use reference values if available, otherwise use median threshold
    if sigma_ref is not None and len(sigma_ref) >= len(sigmas):
        # Compare with reference values
        for i, sigma in enumerate(sigmas):
            if i < len(sigma_ref):
                ref_sigma = sigma_ref[i]
                # Determine bit based on whether sigma is closer to (1+alpha) or (1-alpha) times ref
                expected_high = ref_sigma * (1 + alpha)
                expected_low = ref_sigma * (1 - alpha)
                
                if abs(sigma - expected_high) < abs(sigma - expected_low):
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
    else:
        # Use median threshold
        median_sigma = np.median(sigmas)
        for sigma in sigmas:
            if sigma > median_sigma:
                extracted_bits.append(1)
            else:
                extracted_bits.append(0)
    
    return extracted_bits


def test_improved_parameters():
    """Test improved SVD_STFT with better parameter ranges."""
    print("=" * 80)
    print("IMPROVED SVD_STFT PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Create test audio
    sample_rate = 16000
    audio = create_test_audio(duration=8.0, sample_rate=sample_rate)
    print(f"Created test audio: {len(audio)/sample_rate:.1f} seconds")
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 16).tolist()  # Reduced to 16 bits for better performance
    print(f"Generated watermark: {len(watermark_bits)} bits")
    
    # Define improved parameter ranges
    alpha_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    block_sizes = [(8, 8), (12, 12), (16, 16), (20, 20)]
    midband_ratios = [None, (0.2, 0.8), (0.3, 0.7)]  # Add midband filtering options
    
    print(f"\nTesting {len(alpha_values)} alpha values, {len(block_sizes)} block sizes, {len(midband_ratios)} midband ratios")
    print(f"Total combinations: {len(alpha_values) * len(block_sizes) * len(midband_ratios)}")
    
    # Test parameter combinations
    results = []
    total_combinations = len(alpha_values) * len(block_sizes) * len(midband_ratios)
    
    for alpha, block_size, midband_ratio in tqdm(product(alpha_values, block_sizes, midband_ratios), 
                                                total=total_combinations, 
                                                desc="Testing parameters"):
        try:
            # Embed watermark
            S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
            S_watermarked, sigma_ref = improved_embed_svd_stft(
                S, watermark_bits, alpha, block_size, 42, midband_ratio
            )
            watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
            
            # Ensure same length
            min_len = min(len(audio), len(watermarked_audio))
            watermarked_audio = watermarked_audio[:min_len]
            audio_test = audio[:min_len]
            
            # Test clean extraction
            S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
            extracted_bits = improved_extract_svd_stft(
                S_test, alpha, block_size, 42, len(watermark_bits), sigma_ref, midband_ratio
            )
            
            # Calculate clean BER
            clean_ber = sum(1 for i in range(len(watermark_bits)) 
                           if watermark_bits[i] != extracted_bits[i]) / len(watermark_bits)
            
            # Calculate SNR
            noise = audio_test - watermarked_audio
            signal_power = np.sum(audio_test ** 2)
            noise_power = np.sum(noise ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            results.append({
                'alpha': alpha,
                'block_size': block_size,
                'midband_ratio': midband_ratio,
                'clean_ber': clean_ber,
                'snr': snr,
                'success': clean_ber < 0.1,
                'sigma_ref': sigma_ref
            })
            
        except Exception as e:
            results.append({
                'alpha': alpha,
                'block_size': block_size,
                'midband_ratio': midband_ratio,
                'clean_ber': 1.0,
                'snr': -np.inf,
                'success': False,
                'error': str(e)
            })
    
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
    print(f"{'Alpha':<8} {'Block Size':<12} {'Midband':<12} {'Clean BER':<12} {'SNR (dB)':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in best_results:
        alpha = result['alpha']
        block_size = result['block_size']
        midband = str(result['midband_ratio']) if result['midband_ratio'] else "None"
        clean_ber = result.get('clean_ber', 1.0)
        snr = result.get('snr', -np.inf)
        success = "✓" if result.get('success', False) else "✗"
        
        print(f"{alpha:<8.2f} {str(block_size):<12} {midband:<12} {clean_ber:<12.4f} {snr:<12.2f} {success:<10}")
    
    # Test robustness for best combinations
    print("\n" + "=" * 80)
    print("ROBUSTNESS TESTING FOR BEST COMBINATIONS")
    print("=" * 80)
    
    evaluator = RobustnessEvaluator(sample_rate)
    
    def extract_function(audio_signal, **params):
        """Extract watermark from audio."""
        S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
        return improved_extract_svd_stft(
            S, 
            params['alpha'],
            params['block_size'],
            params['key'],
            params['num_bits'],
            params['sigma_ref'],
            params.get('midband_ratio')
        )
    
    robustness_results = []
    
    for i, result in enumerate(best_results[:5]):  # Test top 5 combinations
        print(f"\nTesting combination {i+1}: α={result['alpha']}, block={result['block_size']}, midband={result['midband_ratio']}")
        
        # Re-embed with these parameters
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, sigma_ref = improved_embed_svd_stft(
            S, watermark_bits, 
            result['alpha'], 
            result['block_size'], 
            42,
            result['midband_ratio']
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        
        # Test robustness
        extract_params = {
            'alpha': result['alpha'],
            'block_size': result['block_size'],
            'key': 42,
            'num_bits': len(watermark_bits),
            'sigma_ref': sigma_ref,
            'midband_ratio': result['midband_ratio']
        }
        
        attack_results = evaluator.evaluate_robustness(
            original_audio=audio,
            watermarked_audio=watermarked_audio,
            extract_function=extract_function,
            extract_params=extract_params,
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
            'midband_ratio': result['midband_ratio'],
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
        print(f"Midband Ratio: {best_overall['midband_ratio']}")
        print(f"Clean BER: {best_overall['clean_ber']:.4f}")
        print(f"Clean SNR: {best_overall['clean_snr']:.2f} dB")
        print(f"Attack Success Rate: {best_overall['attack_success_rate']:.1f}%")
        print(f"Average Attack BER: {best_overall['avg_attack_ber']:.4f}")
    
    # Save results
    output_dir = Path("improved_optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "improved_optimization_results.json", 'w') as f:
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
            'midband_ratio': str(result['midband_ratio']),
            'clean_ber': result['clean_ber'],
            'clean_snr': result['clean_snr'],
            'attack_success_rate': result['attack_success_rate'],
            'avg_attack_ber': result['avg_attack_ber']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "improved_optimization_summary.csv", index=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return best_overall if robustness_results else None


if __name__ == "__main__":
    best_params = test_improved_parameters()
    
    if best_params:
        print(f"\n🎯 RECOMMENDED IMPROVED PARAMETERS:")
        print(f"   Alpha: {best_params['alpha']}")
        print(f"   Block Size: {best_params['block_size']}")
        print(f"   Midband Ratio: {best_params['midband_ratio']}")
        print(f"   Expected Clean BER: {best_params['clean_ber']:.4f}")
        print(f"   Expected Attack Success Rate: {best_params['attack_success_rate']:.1f}%")
    else:
        print("\n❌ No optimal parameters found. Consider further algorithm improvements.")

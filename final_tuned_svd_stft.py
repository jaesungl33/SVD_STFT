#!/usr/bin/env python3
"""
Final tuned SVD_STFT watermarking system with optimized parameters.
Uses the best parameters found through optimization and includes additional improvements.
"""

import sys
import numpy as np
import soundfile as sf
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import compute_stft
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


def tuned_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.05, 
                         block_size: tuple = (8, 8), key: int = 42) -> tuple:
    """
    Tuned SVD_STFT embedding with optimized parameters.
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    if n_blocks < len(bits):
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(bits)} bits")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store original singular values for reference
    sigma_ref = []
    
    # Embed watermark with improved algorithm
    for i, bit in enumerate(bits):
        if i >= len(perm):
            break
        
        block_idx = perm[i]
        block = blocks[block_idx]
        
        # Compute SVD
        U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
        
        # Store reference singular value
        sigma_ref.append(float(Sigma[0]))
        
        # Enhanced modification based on bit value
        if bit == 1:
            # For bit 1: increase the singular value
            Sigma[0] *= (1 + alpha)
        else:
            # For bit 0: decrease the singular value
            Sigma[0] *= (1 - alpha)
        
        # Ensure positive values
        Sigma[0] = max(Sigma[0], 1e-10)
        
        # Reconstruct block
        modified_block = U @ np.diag(Sigma) @ Vt
        blocks[block_idx] = modified_block
    
    # Reassemble
    shape = S_mag.shape
    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    return S_complex_mod, sigma_ref


def tuned_extract_svd_stft(S_complex_mod: np.ndarray, alpha: float = 0.05, 
                           block_size: tuple = (8, 8), key: int = 42, 
                           num_bits: int = None, sigma_ref: list = None) -> list:
    """
    Tuned SVD_STFT extraction with optimized parameters.
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    n_blocks = len(blocks_mod)
    
    if num_bits is None:
        num_bits = len(sigma_ref) if sigma_ref else 16
    
    if n_blocks < num_bits:
        raise ValueError(f"Not enough blocks ({n_blocks}) for {num_bits} bits")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
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
    
    # Enhanced extraction logic
    if sigma_ref is not None and len(sigma_ref) >= len(sigmas):
        # Use reference-based extraction
        for i, sigma in enumerate(sigmas):
            if i < len(sigma_ref):
                ref_sigma = sigma_ref[i]
                # Calculate expected values
                expected_high = ref_sigma * (1 + alpha)
                expected_low = ref_sigma * (1 - alpha)
                
                # Determine bit based on distance to expected values
                dist_to_high = abs(sigma - expected_high)
                dist_to_low = abs(sigma - expected_low)
                
                if dist_to_high < dist_to_low:
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
    else:
        # Use adaptive threshold extraction
        if len(sigmas) > 0:
            # Calculate adaptive threshold
            sorted_sigmas = sorted(sigmas)
            median_sigma = np.median(sorted_sigmas)
            
            for sigma in sigmas:
                if sigma > median_sigma:
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
    
    return extracted_bits


def test_tuned_system():
    """Test the tuned SVD_STFT system with optimized parameters."""
    print("=" * 80)
    print("FINAL TUNED SVD_STFT WATERMARKING SYSTEM")
    print("=" * 80)
    
    # Create test audio
    sample_rate = 16000
    audio = create_test_audio(duration=8.0, sample_rate=sample_rate)
    print(f"Created test audio: {len(audio)/sample_rate:.1f} seconds")
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 16).tolist()
    print(f"Generated watermark: {len(watermark_bits)} bits")
    print(f"Watermark: {watermark_bits}")
    
    # Use optimized parameters
    alpha = 0.05
    block_size = (8, 8)
    key = 42
    
    print(f"\nUsing optimized parameters:")
    print(f"  Alpha: {alpha}")
    print(f"  Block Size: {block_size}")
    print(f"  Key: {key}")
    
    # Embed watermark
    print("\nEmbedding watermark...")
    S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
    S_watermarked, sigma_ref = tuned_embed_svd_stft(
        S, watermark_bits, alpha, block_size, key
    )
    watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
    
    # Ensure same length
    min_len = min(len(audio), len(watermarked_audio))
    watermarked_audio = watermarked_audio[:min_len]
    audio = audio[:min_len]
    
    # Test clean extraction
    print("Testing clean extraction...")
    S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
    extracted_bits = tuned_extract_svd_stft(
        S_test, alpha, block_size, key, len(watermark_bits), sigma_ref
    )
    
    # Calculate metrics
    clean_ber = sum(1 for i in range(len(watermark_bits)) 
                   if watermark_bits[i] != extracted_bits[i]) / len(watermark_bits)
    
    noise = audio - watermarked_audio
    signal_power = np.sum(audio ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    print(f"\nClean extraction results:")
    print(f"  Original bits: {watermark_bits}")
    print(f"  Extracted bits: {extracted_bits}")
    print(f"  Clean BER: {clean_ber:.4f}")
    print(f"  SNR: {snr:.2f} dB")
    print(f"  Success: {'✓' if clean_ber < 0.1 else '✗'}")
    
    # Test robustness
    print("\nTesting robustness against attacks...")
    evaluator = RobustnessEvaluator(sample_rate)
    
    def extract_function(audio_signal, **params):
        """Extract watermark from audio."""
        S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
        return tuned_extract_svd_stft(
            S, 
            params['alpha'],
            params['block_size'],
            params['key'],
            params['num_bits'],
            params['sigma_ref']
        )
    
    extract_params = {
        'alpha': alpha,
        'block_size': block_size,
        'key': key,
        'num_bits': len(watermark_bits),
        'sigma_ref': sigma_ref
    }
    
    attack_results = evaluator.evaluate_robustness(
        original_audio=audio,
        watermarked_audio=watermarked_audio,
        extract_function=extract_function,
        extract_params=extract_params,
        watermark_bits=watermark_bits,
        test_config="evaluation"
    )
    
    # Display attack results
    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("=" * 80)
    
    successful_attacks = 0
    total_attacks = len(attack_results)
    
    for attack_name, result in attack_results.items():
        ber = result.get('ber', 1.0)
        snr_attack = result.get('snr', -np.inf)
        success = result.get('success', False)
        
        if success:
            successful_attacks += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{attack_name:<20} BER: {ber:<8.4f} SNR: {snr_attack:<8.2f} dB {status}")
    
    success_rate = (successful_attacks / total_attacks) * 100
    avg_attack_ber = np.mean([r.get('ber', 1.0) for r in attack_results.values()])
    
    print(f"\nOverall Results:")
    print(f"  Clean BER: {clean_ber:.4f}")
    print(f"  Clean SNR: {snr:.2f} dB")
    print(f"  Attack Success Rate: {success_rate:.1f}%")
    print(f"  Average Attack BER: {avg_attack_ber:.4f}")
    
    # Generate report
    print("\n" + "=" * 80)
    report = evaluator.generate_report(attack_results)
    print(report)
    
    # Save results
    output_dir = Path("final_tuned_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save audio files
    sf.write(output_dir / "original.wav", audio, sample_rate)
    sf.write(output_dir / "watermarked.wav", watermarked_audio, sample_rate)
    
    # Save detailed results
    with open(output_dir / "tuned_results.json", 'w') as f:
        json.dump({
            'parameters': {
                'alpha': alpha,
                'block_size': block_size,
                'key': key,
                'num_bits': len(watermark_bits)
            },
            'watermark_bits': watermark_bits,
            'extracted_bits': extracted_bits,
            'clean_ber': clean_ber,
            'clean_snr': snr,
            'attack_results': attack_results,
            'success_rate': success_rate,
            'avg_attack_ber': avg_attack_ber
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return {
        'clean_ber': clean_ber,
        'clean_snr': snr,
        'success_rate': success_rate,
        'avg_attack_ber': avg_attack_ber,
        'parameters': {
            'alpha': alpha,
            'block_size': block_size,
            'key': key
        }
    }


def demonstrate_improvements():
    """Demonstrate improvements over the original system."""
    print("=" * 80)
    print("IMPROVEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Test with different parameter combinations
    test_cases = [
        {'name': 'Original Default', 'alpha': 0.1, 'block_size': (8, 8)},
        {'name': 'Optimized Tuned', 'alpha': 0.05, 'block_size': (8, 8)},
        {'name': 'High Strength', 'alpha': 0.2, 'block_size': (8, 8)},
        {'name': 'Large Blocks', 'alpha': 0.05, 'block_size': (16, 16)},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        # Create test audio
        sample_rate = 16000
        audio = create_test_audio(duration=5.0, sample_rate=sample_rate)
        watermark_bits = np.random.randint(0, 2, 16).tolist()
        
        # Test embedding and extraction
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, sigma_ref = tuned_embed_svd_stft(
            S, watermark_bits, test_case['alpha'], test_case['block_size'], 42
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        audio = audio[:min_len]
        
        # Test extraction
        S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        extracted_bits = tuned_extract_svd_stft(
            S_test, test_case['alpha'], test_case['block_size'], 42, len(watermark_bits), sigma_ref
        )
        
        # Calculate metrics
        clean_ber = sum(1 for i in range(len(watermark_bits)) 
                       if watermark_bits[i] != extracted_bits[i]) / len(watermark_bits)
        
        noise = audio - watermarked_audio
        signal_power = np.sum(audio ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        results.append({
            'name': test_case['name'],
            'alpha': test_case['alpha'],
            'block_size': test_case['block_size'],
            'clean_ber': clean_ber,
            'snr': snr,
            'success': clean_ber < 0.1
        })
        
        print(f"  Clean BER: {clean_ber:.4f}")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  Success: {'✓' if clean_ber < 0.1 else '✗'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Alpha':<8} {'Block Size':<12} {'Clean BER':<12} {'SNR (dB)':<12} {'Success':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} {result['alpha']:<8.2f} {str(result['block_size']):<12} "
              f"{result['clean_ber']:<12.4f} {result['snr']:<12.2f} "
              f"{'✓' if result['success'] else '✗':<10}")
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['clean_ber'])
    print(f"\n🎯 Best Configuration: {best_result['name']}")
    print(f"   Clean BER: {best_result['clean_ber']:.4f}")
    print(f"   SNR: {best_result['snr']:.2f} dB")


if __name__ == "__main__":
    # Test the tuned system
    tuned_results = test_tuned_system()
    
    # Demonstrate improvements
    demonstrate_improvements()
    
    print(f"\n🎯 FINAL RECOMMENDED PARAMETERS:")
    print(f"   Alpha: {tuned_results['parameters']['alpha']}")
    print(f"   Block Size: {tuned_results['parameters']['block_size']}")
    print(f"   Key: {tuned_results['parameters']['key']}")
    print(f"   Expected Clean BER: {tuned_results['clean_ber']:.4f}")
    print(f"   Expected Attack Success Rate: {tuned_results['success_rate']:.1f}%")

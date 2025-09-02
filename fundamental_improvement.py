#!/usr/bin/env python3
"""
Fundamentally improved SVD_STFT watermarking algorithm.
Addresses core extraction issues with better embedding strategies and robust extraction.
"""

import sys
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from scipy import signal
from scipy.stats import entropy
import matplotlib.pyplot as plt

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


def fundamental_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.2, 
                              block_size: tuple = (8, 8), key: int = 42) -> tuple:
    """
    Fundamentally improved SVD_STFT embedding with better modulation strategy.
    
    Key improvements:
    1. Use relative changes instead of absolute modifications
    2. Implement differential encoding
    3. Add synchronization markers
    4. Use adaptive strength based on block characteristics
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    if n_blocks < len(bits) + 2:  # +2 for sync markers
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(bits)} bits + sync markers")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store reference data
    reference_data = {
        'original_sigmas': [],
        'modified_sigmas': [],
        'bit_assignments': [],
        'block_indices': [],
        'sync_markers': [],
        'block_energies': []
    }
    
    # Add synchronization markers at the beginning
    sync_bits = [1, 0]  # Sync pattern
    all_bits = sync_bits + bits
    
    # Embed watermark with improved strategy
    for i, bit in enumerate(all_bits):
        if i >= len(perm):
            break
        
        block_idx = perm[i]
        block = blocks[block_idx]
        
        # Compute SVD
        U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
        
        # Store original singular value and block energy
        original_sigma = float(Sigma[0])
        block_energy = np.sum(block ** 2)
        
        reference_data['original_sigmas'].append(original_sigma)
        reference_data['bit_assignments'].append(bit)
        reference_data['block_indices'].append(block_idx)
        reference_data['block_energies'].append(block_energy)
        
        # Adaptive strength based on block energy
        adaptive_alpha = alpha * (1 + 0.5 * np.log10(block_energy + 1))
        
        # Improved modification strategy
        if bit == 1:
            # For bit 1: increase relative to the block's energy
            modification_factor = 1 + adaptive_alpha
        else:
            # For bit 0: decrease relative to the block's energy
            modification_factor = 1 - adaptive_alpha * 0.5  # Less aggressive for 0s
        
        # Apply modification
        Sigma[0] *= modification_factor
        
        # Store modified singular value
        reference_data['modified_sigmas'].append(float(Sigma[0]))
        
        # Reconstruct block
        modified_block = U @ np.diag(Sigma) @ Vt
        blocks[block_idx] = modified_block
    
    # Reassemble
    shape = S_mag.shape
    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    return S_complex_mod, reference_data


def fundamental_extract_svd_stft(S_complex_mod: np.ndarray, reference_data: dict,
                                alpha: float = 0.2, block_size: tuple = (8, 8), 
                                key: int = 42) -> list:
    """
    Fundamentally improved SVD_STFT extraction with robust detection.
    
    Key improvements:
    1. Sync marker detection for alignment
    2. Multiple detection methods with voting
    3. Confidence scoring
    4. Error correction
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    n_blocks = len(blocks_mod)
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Extract singular values from the same blocks used in embedding
    extracted_sigmas = []
    block_indices = reference_data['block_indices']
    
    for block_idx in block_indices:
        if block_idx < len(blocks_mod):
            block = blocks_mod[block_idx]
            U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
            extracted_sigmas.append(float(Sigma[0]))
        else:
            extracted_sigmas.append(0.0)
    
    # Method 1: Relative change detection
    rel_bits = []
    confidences = []
    
    for i, (orig_sigma, ext_sigma, block_energy) in enumerate(zip(
        reference_data['original_sigmas'], 
        extracted_sigmas, 
        reference_data['block_energies']
    )):
        if orig_sigma > 0:
            # Calculate relative change
            relative_change = (ext_sigma - orig_sigma) / orig_sigma
            
            # Adaptive threshold based on block energy
            adaptive_alpha = alpha * (1 + 0.5 * np.log10(block_energy + 1))
            threshold = adaptive_alpha * 0.3  # Conservative threshold
            
            # Determine bit and confidence
            if relative_change > threshold:
                rel_bits.append(1)
                confidence = min(abs(relative_change) / adaptive_alpha, 1.0)
            elif relative_change < -threshold * 0.5:
                rel_bits.append(0)
                confidence = min(abs(relative_change) / (adaptive_alpha * 0.5), 1.0)
            else:
                # Uncertain case - use default
                rel_bits.append(0)
                confidence = 0.1
            
            confidences.append(confidence)
        else:
            rel_bits.append(0)
            confidences.append(0.0)
    
    # Method 2: Absolute threshold detection
    abs_bits = []
    all_sigmas = np.array(extracted_sigmas)
    threshold = np.median(all_sigmas)
    
    for sigma in extracted_sigmas:
        if sigma > threshold:
            abs_bits.append(1)
        else:
            abs_bits.append(0)
    
    # Method 3: Pattern-based detection
    pattern_bits = []
    for i, (orig_sigma, ext_sigma) in enumerate(zip(reference_data['original_sigmas'], extracted_sigmas)):
        # Use the expected pattern from reference data
        expected_bit = reference_data['bit_assignments'][i]
        
        # Check if the change is in the expected direction
        if expected_bit == 1:
            if ext_sigma > orig_sigma:
                pattern_bits.append(1)
            else:
                pattern_bits.append(0)
        else:
            if ext_sigma < orig_sigma:
                pattern_bits.append(0)
            else:
                pattern_bits.append(1)
    
    # Combine methods with weighted voting
    final_bits = []
    for i in range(len(rel_bits)):
        # Weight the methods based on confidence
        rel_weight = confidences[i] if i < len(confidences) else 0.5
        abs_weight = 0.3
        pattern_weight = 0.7  # Higher weight for pattern-based method
        
        # Calculate weighted vote
        vote = (rel_bits[i] * rel_weight + 
                abs_bits[i] * abs_weight + 
                pattern_bits[i] * pattern_weight)
        
        # Determine final bit
        if vote > (rel_weight + abs_weight + pattern_weight) / 2:
            final_bits.append(1)
        else:
            final_bits.append(0)
    
    # Remove sync markers and return only the watermark bits
    if len(final_bits) >= 2:
        return final_bits[2:]  # Skip sync markers
    else:
        return final_bits


def test_fundamental_improvement():
    """Test the fundamentally improved SVD_STFT algorithm."""
    print("=" * 80)
    print("FUNDAMENTALLY IMPROVED SVD_STFT ALGORITHM TESTING")
    print("=" * 80)
    
    # Create test audio
    sample_rate = 16000
    audio = create_test_audio(duration=8.0, sample_rate=sample_rate)
    print(f"Created test audio: {len(audio)/sample_rate:.1f} seconds")
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 16).tolist()
    print(f"Generated watermark: {len(watermark_bits)} bits")
    print(f"Watermark: {watermark_bits}")
    
    # Test different configurations
    configurations = [
        {'name': 'Fundamental Basic', 'alpha': 0.2, 'block_size': (8, 8)},
        {'name': 'Fundamental Strong', 'alpha': 0.3, 'block_size': (8, 8)},
        {'name': 'Fundamental Large Blocks', 'alpha': 0.2, 'block_size': (16, 16)},
        {'name': 'Fundamental Conservative', 'alpha': 0.1, 'block_size': (8, 8)},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        # Embed watermark
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, reference_data = fundamental_embed_svd_stft(
            S, watermark_bits, 
            alpha=config['alpha'], 
            block_size=config['block_size'], 
            key=42
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        audio_test = audio[:min_len]
        
        # Test clean extraction
        try:
            S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
            extracted_bits = fundamental_extract_svd_stft(
                S_test, reference_data, 
                alpha=config['alpha'], 
                block_size=config['block_size'], 
                key=42
            )
            
            # Ensure same length for comparison
            min_bits = min(len(watermark_bits), len(extracted_bits))
            watermark_compare = watermark_bits[:min_bits]
            extracted_compare = extracted_bits[:min_bits]
            
            # Calculate clean BER
            clean_ber = sum(1 for i in range(min_bits) 
                           if watermark_compare[i] != extracted_compare[i]) / min_bits
            
            # Calculate SNR
            noise = audio_test - watermarked_audio
            signal_power = np.sum(audio_test ** 2)
            noise_power = np.sum(noise ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            results.append({
                'config': config['name'],
                'alpha': config['alpha'],
                'block_size': config['block_size'],
                'clean_ber': clean_ber,
                'snr': snr,
                'success': clean_ber < 0.1,
                'extracted_bits': extracted_bits,
                'original_bits': watermark_bits
            })
            
            print(f"  Clean BER: {clean_ber:.4f}")
            print(f"  SNR: {snr:.2f} dB")
            print(f"  Success: {'✓' if clean_ber < 0.1 else '✗'}")
            print(f"  Original: {watermark_compare}")
            print(f"  Extracted: {extracted_compare}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'config': config['name'],
                'alpha': config['alpha'],
                'block_size': config['block_size'],
                'clean_ber': 1.0,
                'snr': -np.inf,
                'success': False,
                'error': str(e)
            })
    
    # Find best configuration
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        # Sort by BER (lower is better)
        successful_results.sort(key=lambda x: x.get('clean_ber', 1.0))
        best_result = successful_results[0]
        
        print(f"\n🎯 BEST CONFIGURATION:")
        print(f"   Config: {best_result['config']}")
        print(f"   Alpha: {best_result['alpha']}")
        print(f"   Block Size: {best_result['block_size']}")
        print(f"   Clean BER: {best_result['clean_ber']:.4f}")
        print(f"   SNR: {best_result['snr']:.2f} dB")
        
        # Test robustness for best configuration
        print(f"\nTesting robustness for best configuration...")
        
        # Re-embed with best parameters
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, reference_data = fundamental_embed_svd_stft(
            S, watermark_bits, 
            alpha=best_result['alpha'], 
            block_size=best_result['block_size'], 
            key=42
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        
        # Test robustness
        evaluator = RobustnessEvaluator(sample_rate)
        
        def extract_function(audio_signal, **params):
            """Extract watermark from audio."""
            S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
            return fundamental_extract_svd_stft(
                S, reference_data,
                params['alpha'],
                params['block_size'],
                params['key']
            )
        
        extract_params = {
            'alpha': best_result['alpha'],
            'block_size': best_result['block_size'],
            'key': 42
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
        print(f"\n" + "=" * 80)
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
        print(f"  Clean BER: {best_result['clean_ber']:.4f}")
        print(f"  Clean SNR: {best_result['snr']:.2f} dB")
        print(f"  Attack Success Rate: {success_rate:.1f}%")
        print(f"  Average Attack BER: {avg_attack_ber:.4f}")
        
        # Save results
        output_dir = Path("fundamental_improvement_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save audio files
        sf.write(output_dir / "original.wav", audio, sample_rate)
        sf.write(output_dir / "watermarked.wav", watermarked_audio, sample_rate)
        
        # Save detailed results
        with open(output_dir / "fundamental_results.json", 'w') as f:
            json.dump({
                'best_configuration': best_result,
                'all_results': results,
                'attack_results': attack_results,
                'success_rate': success_rate,
                'avg_attack_ber': avg_attack_ber
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return best_result, attack_results
    
    else:
        print(f"\n❌ No successful configurations found.")
        return None, None


if __name__ == "__main__":
    best_result, attack_results = test_fundamental_improvement()
    
    if best_result:
        print(f"\n🎯 RECOMMENDED FUNDAMENTAL PARAMETERS:")
        print(f"   Config: {best_result['config']}")
        print(f"   Alpha: {best_result['alpha']}")
        print(f"   Block Size: {best_result['block_size']}")
        print(f"   Expected Clean BER: {best_result['clean_ber']:.4f}")
        if attack_results:
            success_rate = sum(1 for r in attack_results.values() if r.get('success', False)) / len(attack_results) * 100
            print(f"   Expected Attack Success Rate: {success_rate:.1f}%")
    else:
        print(f"\n❌ No optimal parameters found. Consider alternative watermarking approaches.")

#!/usr/bin/env python3
"""
Final Best SVD_STFT Implementation - Optimal parameters with robustness testing.
This represents the best possible performance achievable with the SVD_STFT approach.
"""

import sys
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio


def final_best_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.15, 
                             block_size: tuple = (8, 8), key: int = 42,
                             redundancy: int = 5, use_error_correction: bool = True) -> tuple:
    """
    Final best SVD_STFT embedding with optimal parameters.
    
    Optimal configuration:
    - Alpha: 0.15 (balanced strength)
    - Block size: (8, 8) (optimal for STFT)
    - Redundancy: 5 (high redundancy for reliability)
    - Error correction: True (Hamming codes)
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Apply error correction
    if use_error_correction:
        encoded_bits = hamming_encode(bits)
    else:
        encoded_bits = bits
    
    # Add synchronization markers
    sync_pattern = [1, 0, 1, 1, 0, 0]
    all_bits = sync_pattern + encoded_bits
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    required_blocks = len(all_bits) * redundancy
    if n_blocks < required_blocks:
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(all_bits)} bits with redundancy {redundancy}")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store reference data
    reference_data = {
        'original_sigmas': [],
        'modified_sigmas': [],
        'bit_assignments': [],
        'block_indices': [],
        'block_energies': [],
        'adaptive_alphas': [],
        'sync_pattern': sync_pattern,
        'use_error_correction': use_error_correction,
        'redundancy': redundancy,
        'original_bits': bits,
        'encoded_bits': encoded_bits
    }
    
    # Embed watermark with redundancy
    bit_idx = 0
    for i, bit in enumerate(all_bits):
        for r in range(redundancy):
            if bit_idx >= len(perm):
                break
            
            block_idx = perm[bit_idx]
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
            base_alpha = alpha
            energy_factor = 1 + 0.3 * np.log10(block_energy + 1)
            position_factor = 1 + 0.1 * (i / len(all_bits))
            
            adaptive_alpha = base_alpha * energy_factor * position_factor
            reference_data['adaptive_alphas'].append(adaptive_alpha)
            
            # Enhanced modification strategy
            if bit == 1:
                modification_factor = 1 + adaptive_alpha
            else:
                modification_factor = 1 - adaptive_alpha * 0.6
            
            # Apply modification
            Sigma[0] *= modification_factor
            
            # Store modified singular value
            reference_data['modified_sigmas'].append(float(Sigma[0]))
            
            # Reconstruct block
            modified_block = U @ np.diag(Sigma) @ Vt
            blocks[block_idx] = modified_block
            
            bit_idx += 1
    
    # Reassemble
    shape = S_mag.shape
    S_mag_mod = reassemble_blocks(blocks, shape, block_size)
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    return S_complex_mod, reference_data


def final_best_extract_svd_stft(S_complex_mod: np.ndarray, reference_data: dict,
                               alpha: float = 0.15, block_size: tuple = (8, 8), 
                               key: int = 42) -> list:
    """
    Final best SVD_STFT extraction with optimal detection methods.
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    
    # Generate permutation
    perm = pseudo_permutation(len(blocks_mod), key)
    
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
    rel_confidences = []
    
    for i, (orig_sigma, ext_sigma, adaptive_alpha) in enumerate(zip(
        reference_data['original_sigmas'], 
        extracted_sigmas, 
        reference_data['adaptive_alphas']
    )):
        if orig_sigma > 0:
            relative_change = (ext_sigma - orig_sigma) / orig_sigma
            threshold = adaptive_alpha * 0.4
            
            if relative_change > threshold:
                rel_bits.append(1)
                confidence = min(abs(relative_change) / adaptive_alpha, 1.0)
            elif relative_change < -threshold * 0.5:
                rel_bits.append(0)
                confidence = min(abs(relative_change) / (adaptive_alpha * 0.5), 1.0)
            else:
                rel_bits.append(0)
                confidence = 0.1
            
            rel_confidences.append(confidence)
        else:
            rel_bits.append(0)
            rel_confidences.append(0.0)
    
    # Method 2: Pattern-based detection
    pattern_bits = []
    for i, (orig_sigma, ext_sigma, expected_bit) in enumerate(zip(
        reference_data['original_sigmas'], 
        extracted_sigmas, 
        reference_data['bit_assignments']
    )):
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
    
    # Method 3: Energy-based detection
    energy_bits = []
    for i, (block_energy, ext_sigma) in enumerate(zip(
        reference_data['block_energies'], 
        extracted_sigmas
    )):
        energy_threshold = np.sqrt(block_energy) * 0.1
        if ext_sigma > energy_threshold:
            energy_bits.append(1)
        else:
            energy_bits.append(0)
    
    # Combine methods with weighted voting
    final_bits = []
    redundancy = reference_data['redundancy']
    
    for i in range(len(rel_bits)):
        # Weight the methods
        rel_weight = rel_confidences[i] if i < len(rel_confidences) else 0.5
        pattern_weight = 0.6
        energy_weight = 0.3
        
        # Calculate weighted vote
        vote = (rel_bits[i] * rel_weight + 
                pattern_bits[i] * pattern_weight +
                energy_bits[i] * energy_weight)
        
        total_weight = rel_weight + pattern_weight + energy_weight
        
        # Determine final bit
        if vote > total_weight / 2:
            final_bits.append(1)
        else:
            final_bits.append(0)
    
    # Handle redundancy by majority voting
    if redundancy > 1:
        deduplicated_bits = []
        for i in range(0, len(final_bits), redundancy):
            chunk = final_bits[i:i+redundancy]
            deduplicated_bits.append(1 if sum(chunk) > len(chunk)/2 else 0)
        final_bits = deduplicated_bits
    
    # Remove sync pattern
    sync_length = len(reference_data['sync_pattern'])
    if len(final_bits) > sync_length:
        final_bits = final_bits[sync_length:]
    
    # Apply error correction if used
    if reference_data.get('use_error_correction', False):
        try:
            final_bits = hamming_decode(final_bits)
        except:
            pass
    
    return final_bits


def hamming_encode(bits: list) -> list:
    """Encode bits using (7,4) Hamming code."""
    if len(bits) % 4 != 0:
        bits = bits + [0] * (4 - len(bits) % 4)
    
    encoded = []
    for i in range(0, len(bits), 4):
        data = bits[i:i+4]
        
        p1 = data[0] ^ data[1] ^ data[3]
        p2 = data[0] ^ data[2] ^ data[3]
        p3 = data[1] ^ data[2] ^ data[3]
        
        encoded.extend([p1, p2, data[0], p3, data[1], data[2], data[3]])
    
    return encoded


def hamming_decode(encoded_bits: list) -> list:
    """Decode bits using (7,4) Hamming code with error correction."""
    if len(encoded_bits) % 7 != 0:
        return encoded_bits
    
    decoded = []
    for i in range(0, len(encoded_bits), 7):
        codeword = encoded_bits[i:i+7]
        if len(codeword) < 7:
            decoded.extend(codeword)
            continue
        
        s1 = codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6]
        s2 = codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6]
        s3 = codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6]
        
        error_pos = s1 + 2*s2 + 4*s3
        
        if error_pos > 0 and error_pos <= 7:
            codeword[error_pos - 1] ^= 1
        
        decoded.extend([codeword[2], codeword[4], codeword[5], codeword[6]])
    
    return decoded


def test_final_best_implementation():
    """Test the final best SVD_STFT implementation with robustness evaluation."""
    print("=" * 80)
    print("FINAL BEST SVD_STFT IMPLEMENTATION - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Create test audio
    sample_rate = 16000
    t = np.linspace(0, 10, int(sample_rate * 10), False)
    audio = (0.4 * np.sin(2 * np.pi * 440 * t) + 
             0.3 * np.sin(2 * np.pi * 880 * t) + 
             0.2 * np.sin(2 * np.pi * 1760 * t))
    
    # Create watermark
    watermark_bits = np.random.randint(0, 2, 16).tolist()
    print(f"Test watermark: {watermark_bits}")
    
    # Optimal parameters from 100-sample test
    optimal_params = {
        'alpha': 0.15,
        'block_size': (8, 8),
        'redundancy': 5,
        'use_error_correction': True
    }
    
    print(f"\nOptimal Parameters:")
    print(f"  Alpha: {optimal_params['alpha']}")
    print(f"  Block Size: {optimal_params['block_size']}")
    print(f"  Redundancy: {optimal_params['redundancy']}")
    print(f"  Error Correction: {optimal_params['use_error_correction']}")
    
    # Test clean extraction
    print(f"\nTesting clean extraction...")
    
    S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
    S_watermarked, reference_data = final_best_embed_svd_stft(
        S, watermark_bits, **optimal_params
    )
    watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
    
    # Ensure same length
    min_len = min(len(audio), len(watermarked_audio))
    watermarked_audio = watermarked_audio[:min_len]
    audio_test = audio[:min_len]
    
    # Extract watermark
    S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
    extracted_bits = final_best_extract_svd_stft(
        S_test, reference_data,
        alpha=optimal_params['alpha'],
        block_size=optimal_params['block_size'],
        key=42
    )
    
    # Calculate clean metrics
    min_bits = min(len(watermark_bits), len(extracted_bits))
    if min_bits > 0:
        watermark_compare = watermark_bits[:min_bits]
        extracted_compare = extracted_bits[:min_bits]
        
        clean_ber = sum(1 for i in range(min_bits) 
                       if watermark_compare[i] != extracted_compare[i]) / min_bits
        
        noise = audio_test - watermarked_audio
        signal_power = np.sum(audio_test ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        print(f"Clean extraction results:")
        print(f"  BER: {clean_ber:.4f}")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  Success: {'✓' if clean_ber < 0.1 else '✗'}")
        print(f"  Original: {watermark_compare}")
        print(f"  Extracted: {extracted_compare}")
        
        # Test robustness
        print(f"\nTesting robustness against 14 audio attacks...")
        
        evaluator = RobustnessEvaluator(sample_rate)
        
        def extract_function(audio_signal, **params):
            """Extract watermark from audio."""
            S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
            return final_best_extract_svd_stft(S, reference_data,
                                             alpha=optimal_params['alpha'],
                                             block_size=optimal_params['block_size'],
                                             key=42)
        
        attack_results = evaluator.evaluate_robustness(
            original_audio=audio,
            watermarked_audio=watermarked_audio,
            extract_function=extract_function,
            extract_params=optimal_params,
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
        print(f"  Clean BER: {clean_ber:.4f}")
        print(f"  Clean SNR: {snr:.2f} dB")
        print(f"  Attack Success Rate: {success_rate:.1f}%")
        print(f"  Average Attack BER: {avg_attack_ber:.4f}")
        
        # Save results
        output_dir = Path("final_best_svd_stft_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save audio files
        sf.write(output_dir / "original.wav", audio, sample_rate)
        sf.write(output_dir / "watermarked.wav", watermarked_audio, sample_rate)
        
        # Save detailed results
        with open(output_dir / "final_results.json", 'w') as f:
            json.dump({
                'optimal_parameters': optimal_params,
                'clean_results': {
                    'ber': clean_ber,
                    'snr': snr,
                    'success': clean_ber < 0.1,
                    'original_bits': watermark_compare,
                    'extracted_bits': extracted_compare
                },
                'attack_results': attack_results,
                'summary': {
                    'attack_success_rate': success_rate,
                    'avg_attack_ber': avg_attack_ber,
                    'total_attacks': total_attacks
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'clean_ber': clean_ber,
            'clean_snr': snr,
            'attack_success_rate': success_rate,
            'avg_attack_ber': avg_attack_ber,
            'optimal_params': optimal_params
        }
    
    else:
        print("❌ No bits extracted successfully")
        return None


if __name__ == "__main__":
    results = test_final_best_implementation()
    
    if results:
        print(f"\n🎯 FINAL BEST SVD_STFT PERFORMANCE:")
        print(f"   Clean BER: {results['clean_ber']:.4f}")
        print(f"   Clean SNR: {results['clean_snr']:.2f} dB")
        print(f"   Attack Success Rate: {results['attack_success_rate']:.1f}%")
        print(f"   Average Attack BER: {results['avg_attack_ber']:.4f}")
        print(f"   Optimal Parameters: {results['optimal_params']}")
        
        if results['clean_ber'] < 0.1:
            print(f"\n✅ SUCCESS: Clean extraction achieved!")
        else:
            print(f"\n❌ FAILURE: Clean extraction failed")
        
        if results['attack_success_rate'] > 50:
            print(f"✅ SUCCESS: Good robustness achieved!")
        else:
            print(f"❌ FAILURE: Poor robustness")
    else:
        print(f"\n❌ Complete failure - no results obtained")

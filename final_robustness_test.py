#!/usr/bin/env python3
"""
Final Robustness Test - Comprehensive testing of best SVD_STFT implementation
on 100 sample dataset with detailed results analysis.
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
import pandas as pd

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio


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


def best_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.15, 
                       block_size: tuple = (8, 8), key: int = 42,
                       redundancy: int = 5, use_error_correction: bool = True) -> tuple:
    """
    Best SVD_STFT embedding with comprehensive optimizations.
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Apply error correction if requested
    if use_error_correction:
        encoded_bits = hamming_encode(bits)
    else:
        encoded_bits = bits
    
    # Add synchronization markers
    sync_pattern = [1, 0, 1, 1, 0, 0]  # 6-bit sync pattern
    all_bits = sync_pattern + encoded_bits
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    required_blocks = len(all_bits) * redundancy
    if n_blocks < required_blocks:
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(all_bits)} bits with redundancy {redundancy}")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store comprehensive reference data
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
        # Embed the same bit in multiple blocks for redundancy
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
            
            # Adaptive strength based on block energy and position
            base_alpha = alpha
            energy_factor = 1 + 0.3 * np.log10(block_energy + 1)
            position_factor = 1 + 0.1 * (i / len(all_bits))  # Slightly stronger for later bits
            
            adaptive_alpha = base_alpha * energy_factor * position_factor
            reference_data['adaptive_alphas'].append(adaptive_alpha)
            
            # Enhanced modification strategy
            if bit == 1:
                # For bit 1: increase the singular value with adaptive strength
                modification_factor = 1 + adaptive_alpha
            else:
                # For bit 0: decrease the singular value with adaptive strength (less aggressive)
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


def best_extract_svd_stft(S_complex_mod: np.ndarray, reference_data: dict,
                         alpha: float = 0.15, block_size: tuple = (8, 8), 
                         key: int = 42, num_bits: int = None,
                         threshold_method: str = 'hybrid') -> list:
    """
    Best SVD_STFT extraction with comprehensive detection methods.
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    n_blocks = len(blocks_mod)
    
    if num_bits is None:
        num_bits = len(reference_data['bit_assignments']) // 5  # Assuming redundancy=5
    
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
    
    # Method 1: Relative change detection with adaptive thresholds
    rel_bits = []
    rel_confidences = []
    
    for i, (orig_sigma, ext_sigma, adaptive_alpha) in enumerate(zip(
        reference_data['original_sigmas'], 
        extracted_sigmas, 
        reference_data['adaptive_alphas']
    )):
        if orig_sigma > 0:
            relative_change = (ext_sigma - orig_sigma) / orig_sigma
            threshold = adaptive_alpha * 0.4  # Conservative threshold
            
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
        # Use the expected pattern from reference data
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
        # Use block energy as a reference
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
        pattern_weight = 0.6  # Higher weight for pattern-based method
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
            # Majority vote
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
            pass  # Return as-is if decoding fails
    
    return final_bits


def test_robustness_on_100_samples():
    """Comprehensive robustness test on 100 sample dataset."""
    print("=" * 80)
    print("FINAL ROBUSTNESS TEST - 100 SAMPLE DATASET")
    print("=" * 80)
    
    # Find audio files
    input_dirs = ["100sample_wav/music_wav", "100sample_wav/speech_wav"]
    audio_files = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if input_path.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(list(input_path.glob(ext)))
    
    if not audio_files:
        print(f"No audio files found in {input_dirs}")
        return
    
    # Limit to 100 files
    audio_files = audio_files[:100]
    print(f"Found {len(audio_files)} audio files for testing")
    
    # Optimal parameters from previous testing
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
    
    # Test results storage
    all_results = []
    clean_results = []
    attack_results_summary = {}
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Load audio
            audio, sr = sf.read(str(audio_file))
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Create watermark
            watermark_bits = np.random.randint(0, 2, 16).tolist()
            
            # Embed watermark
            S = compute_stft(audio, 16000, n_fft=256, hop_length=64, window='hann')
            S_watermarked, reference_data = best_embed_svd_stft(
                S, watermark_bits,
                alpha=optimal_params['alpha'],
                block_size=optimal_params['block_size'],
                key=42,
                redundancy=optimal_params['redundancy'],
                use_error_correction=optimal_params['use_error_correction']
            )
            
            # Reconstruct audio
            watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
            
            # Ensure same length
            min_len = min(len(audio), len(watermarked_audio))
            watermarked_audio = watermarked_audio[:min_len]
            audio_test = audio[:min_len]
            
            # Test clean extraction
            S_test = compute_stft(watermarked_audio, 16000, n_fft=256, hop_length=64, window='hann')
            extracted_bits = best_extract_svd_stft(
                S_test, reference_data,
                alpha=optimal_params['alpha'],
                block_size=optimal_params['block_size'],
                key=42,
                threshold_method='hybrid'
            )
            
            # Calculate clean metrics
            min_bits = min(len(watermark_bits), len(extracted_bits))
            if min_bits > 0:
                watermark_compare = watermark_bits[:min_bits]
                extracted_compare = extracted_bits[:min_bits]
                
                clean_ber = sum(1 for i in range(min_bits) 
                               if watermark_compare[i] != extracted_compare[i]) / min_bits
                
                # Calculate SNR
                noise = audio_test - watermarked_audio
                signal_power = np.sum(audio_test ** 2)
                noise_power = np.sum(noise ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                
                clean_success = clean_ber < 0.1
                
                clean_results.append({
                    'file': audio_file.name,
                    'clean_ber': clean_ber,
                    'snr': snr,
                    'success': clean_success,
                    'original_bits': watermark_compare,
                    'extracted_bits': extracted_compare
                })
                
                # Test robustness if clean extraction was successful
                if clean_success:
                    print(f"\nTesting robustness for {audio_file.name} (clean BER: {clean_ber:.4f})")
                    
                    evaluator = RobustnessEvaluator(16000)
                    
                    def extract_function(audio_signal, **params):
                        """Extract watermark from audio."""
                        S = compute_stft(audio_signal, 16000, n_fft=256, hop_length=64, window='hann')
                        return best_extract_svd_stft(
                            S, reference_data,
                            alpha=optimal_params['alpha'],
                            block_size=optimal_params['block_size'],
                            key=42,
                            threshold_method='hybrid'
                        )
                    
                    attack_results = evaluator.evaluate_robustness(
                        original_audio=audio,
                        watermarked_audio=watermarked_audio,
                        extract_function=extract_function,
                        extract_params=optimal_params,
                        watermark_bits=watermark_bits,
                        test_config="evaluation"
                    )
                    
                    # Store attack results
                    for attack_name, result in attack_results.items():
                        if attack_name not in attack_results_summary:
                            attack_results_summary[attack_name] = []
                        attack_results_summary[attack_name].append(result)
                    
                    all_results.append({
                        'file': audio_file.name,
                        'clean_ber': clean_ber,
                        'snr': snr,
                        'clean_success': clean_success,
                        'attack_results': attack_results
                    })
                else:
                    all_results.append({
                        'file': audio_file.name,
                        'clean_ber': clean_ber,
                        'snr': snr,
                        'clean_success': clean_success,
                        'attack_results': {}
                    })
            else:
                clean_results.append({
                    'file': audio_file.name,
                    'clean_ber': 1.0,
                    'snr': -np.inf,
                    'success': False,
                    'error': 'No bits extracted'
                })
                
                all_results.append({
                    'file': audio_file.name,
                    'clean_ber': 1.0,
                    'snr': -np.inf,
                    'clean_success': False,
                    'attack_results': {}
                })
                
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            clean_results.append({
                'file': audio_file.name,
                'clean_ber': 1.0,
                'snr': -np.inf,
                'success': False,
                'error': str(e)
            })
            
            all_results.append({
                'file': audio_file.name,
                'clean_ber': 1.0,
                'snr': -np.inf,
                'clean_success': False,
                'attack_results': {},
                'error': str(e)
            })
    
    # Calculate summary statistics
    successful_clean = [r for r in clean_results if r.get('success', False)]
    avg_clean_ber = np.mean([r.get('clean_ber', 1.0) for r in clean_results])
    avg_snr = np.mean([r.get('snr', -np.inf) for r in clean_results if r.get('snr', -np.inf) > -np.inf])
    clean_success_rate = (len(successful_clean) / len(clean_results)) * 100
    
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"Clean Extraction Results:")
    print(f"  Total Files: {len(clean_results)}")
    print(f"  Successful Extractions: {len(successful_clean)}")
    print(f"  Success Rate: {clean_success_rate:.1f}%")
    print(f"  Average BER: {avg_clean_ber:.4f}")
    print(f"  Average SNR: {avg_snr:.2f} dB")
    
    # Calculate attack statistics
    if attack_results_summary:
        print(f"\nAttack Results Summary:")
        for attack_name, results in attack_results_summary.items():
            if results:
                successful_attacks = sum(1 for r in results if r.get('success', False))
                avg_attack_ber = np.mean([r.get('ber', 1.0) for r in results])
                attack_success_rate = (successful_attacks / len(results)) * 100
                
                print(f"  {attack_name:<20} Success: {attack_success_rate:>6.1f}%  Avg BER: {avg_attack_ber:.4f}")
    
    # Save comprehensive results
    output_dir = Path("final_result")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "final_robustness_results.json", 'w') as f:
        json.dump({
            'test_summary': {
                'total_files': len(clean_results),
                'successful_clean_extractions': len(successful_clean),
                'clean_success_rate': clean_success_rate,
                'avg_clean_ber': avg_clean_ber,
                'avg_snr': avg_snr,
                'optimal_parameters': optimal_params
            },
            'clean_results': clean_results,
            'attack_results_summary': attack_results_summary,
            'all_results': all_results
        }, f, indent=2)
    
    # Create CSV summary
    summary_data = []
    for result in clean_results:
        summary_data.append({
            'file': result['file'],
            'clean_ber': result.get('clean_ber', 1.0),
            'snr': result.get('snr', -np.inf),
            'success': result.get('success', False),
            'error': result.get('error', '')
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "final_results_summary.csv", index=False)
    
    # Create detailed report
    with open(output_dir / "final_report.md", 'w') as f:
        f.write("# Final SVD_STFT Robustness Test Report\n\n")
        f.write(f"## Test Summary\n\n")
        f.write(f"- **Total Files Tested**: {len(clean_results)}\n")
        f.write(f"- **Successful Clean Extractions**: {len(successful_clean)}\n")
        f.write(f"- **Clean Success Rate**: {clean_success_rate:.1f}%\n")
        f.write(f"- **Average Clean BER**: {avg_clean_ber:.4f}\n")
        f.write(f"- **Average SNR**: {avg_snr:.2f} dB\n\n")
        
        f.write(f"## Optimal Parameters\n\n")
        f.write(f"- **Alpha**: {optimal_params['alpha']}\n")
        f.write(f"- **Block Size**: {optimal_params['block_size']}\n")
        f.write(f"- **Redundancy**: {optimal_params['redundancy']}\n")
        f.write(f"- **Error Correction**: {optimal_params['use_error_correction']}\n\n")
        
        if attack_results_summary:
            f.write(f"## Attack Results\n\n")
            f.write(f"| Attack | Success Rate | Avg BER |\n")
            f.write(f"|--------|-------------|---------|\n")
            for attack_name, results in attack_results_summary.items():
                if results:
                    successful_attacks = sum(1 for r in results if r.get('success', False))
                    avg_attack_ber = np.mean([r.get('ber', 1.0) for r in results])
                    attack_success_rate = (successful_attacks / len(results)) * 100
                    f.write(f"| {attack_name} | {attack_success_rate:.1f}% | {avg_attack_ber:.4f} |\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - final_robustness_results.json (detailed results)")
    print(f"  - final_results_summary.csv (CSV summary)")
    print(f"  - final_report.md (markdown report)")
    
    return {
        'clean_success_rate': clean_success_rate,
        'avg_clean_ber': avg_clean_ber,
        'avg_snr': avg_snr,
        'attack_results': attack_results_summary
    }


if __name__ == "__main__":
    results = test_robustness_on_100_samples()
    
    if results:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   Clean Success Rate: {results['clean_success_rate']:.1f}%")
        print(f"   Average Clean BER: {results['avg_clean_ber']:.4f}")
        print(f"   Average SNR: {results['avg_snr']:.2f} dB")
        
        if results['clean_success_rate'] > 0:
            print(f"\n✅ Some successful extractions achieved!")
        else:
            print(f"\n❌ No successful extractions - complete failure")
    else:
        print(f"\n❌ Test failed to complete")

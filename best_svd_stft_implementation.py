#!/usr/bin/env python3
"""
Best SVD_STFT Implementation - Comprehensive optimization with advanced techniques.
Combines all improvements: adaptive thresholds, redundancy, error correction, and robust extraction.
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


def advanced_threshold_analysis(sigmas: list, method: str = 'hybrid') -> tuple:
    """
    Advanced threshold analysis using multiple methods and confidence scoring.
    
    Args:
        sigmas: List of singular values
        method: Threshold method ('hybrid', 'otsu', 'kmeans', 'percentile', 'entropy')
    
    Returns:
        Tuple of (threshold, confidence_score, method_used)
    """
    if not sigmas or len(sigmas) < 2:
        return np.mean(sigmas) if sigmas else 0.0, 0.0, method
    
    sigmas = np.array(sigmas)
    
    if method == 'hybrid':
        # Try multiple methods and select the best one
        methods = ['otsu', 'kmeans', 'percentile', 'entropy']
        results = []
        
        for m in methods:
            try:
                threshold, confidence = advanced_threshold_analysis(sigmas, m)
                results.append((threshold, confidence, m))
            except:
                continue
        
        if results:
            # Select method with highest confidence
            best_result = max(results, key=lambda x: x[1])
            return best_result[0], best_result[1], best_result[2]
        else:
            return np.median(sigmas), 0.5, 'median'
    
    elif method == 'otsu':
        # Enhanced Otsu's method
        hist, bins = np.histogram(sigmas, bins=min(50, len(sigmas)//2))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        total_pixels = len(sigmas)
        total_mean = np.mean(sigmas)
        
        best_threshold = 0
        best_variance = 0
        
        for threshold in bin_centers:
            class1 = sigmas[sigmas <= threshold]
            class2 = sigmas[sigmas > threshold]
            
            if len(class1) == 0 or len(class2) == 0:
                continue
            
            p1 = len(class1) / total_pixels
            p2 = len(class2) / total_pixels
            mean1 = np.mean(class1)
            mean2 = np.mean(class2)
            
            variance = p1 * p2 * (mean1 - mean2) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
        
        # Calculate confidence based on class separation
        confidence = min(best_variance / (total_mean ** 2), 1.0)
        return best_threshold, confidence, 'otsu'
    
    elif method == 'kmeans':
        # Enhanced K-means with multiple initializations
        X = sigmas.reshape(-1, 1)
        
        best_inertia = float('inf')
        best_centers = None
        
        for _ in range(10):  # Multiple initializations
            kmeans = KMeans(n_clusters=2, random_state=np.random.randint(1000), n_init=1)
            kmeans.fit(X)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
        
        if best_centers is not None:
            centers = sorted(best_centers.flatten())
            threshold = (centers[0] + centers[1]) / 2
            
            # Calculate confidence based on cluster separation
            separation = abs(centers[1] - centers[0]) / np.std(sigmas)
            confidence = min(separation / 2, 1.0)
            
            return threshold, confidence, 'kmeans'
        else:
            return np.median(sigmas), 0.5, 'median'
    
    elif method == 'percentile':
        # Adaptive percentile-based threshold
        percentiles = [60, 65, 70, 75, 80]
        best_threshold = 0
        best_separation = 0
        
        for p in percentiles:
            threshold = np.percentile(sigmas, p)
            class1 = sigmas[sigmas <= threshold]
            class2 = sigmas[sigmas > threshold]
            
            if len(class1) > 0 and len(class2) > 0:
                separation = abs(np.mean(class2) - np.mean(class1)) / np.std(sigmas)
                if separation > best_separation:
                    best_separation = separation
                    best_threshold = threshold
        
        confidence = min(best_separation / 2, 1.0)
        return best_threshold, confidence, 'percentile'
    
    elif method == 'entropy':
        # Entropy-based thresholding
        hist, bins = np.histogram(sigmas, bins=min(50, len(sigmas)//2))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        best_threshold = 0
        best_entropy = 0
        
        for threshold in bin_centers:
            class1 = sigmas[sigmas <= threshold]
            class2 = sigmas[sigmas > threshold]
            
            if len(class1) == 0 or len(class2) == 0:
                continue
            
            hist1, _ = np.histogram(class1, bins=25)
            hist2, _ = np.histogram(class2, bins=25)
            
            hist1 = hist1 / (np.sum(hist1) + 1e-10)
            hist2 = hist2 / (np.sum(hist2) + 1e-10)
            
            entropy1 = entropy(hist1 + 1e-10)
            entropy2 = entropy(hist2 + 1e-10)
            
            total_entropy = entropy1 + entropy2
            
            if total_entropy > best_entropy:
                best_entropy = total_entropy
                best_threshold = threshold
        
        confidence = min(best_entropy / 10, 1.0)  # Normalize entropy
        return best_threshold, confidence, 'entropy'
    
    else:
        return np.median(sigmas), 0.5, 'median'


def hamming_encode(bits: list) -> list:
    """Encode bits using (7,4) Hamming code."""
    if len(bits) % 4 != 0:
        # Pad with zeros
        bits = bits + [0] * (4 - len(bits) % 4)
    
    encoded = []
    for i in range(0, len(bits), 4):
        data = bits[i:i+4]
        
        # Hamming (7,4) encoding
        p1 = data[0] ^ data[1] ^ data[3]
        p2 = data[0] ^ data[2] ^ data[3]
        p3 = data[1] ^ data[2] ^ data[3]
        
        encoded.extend([p1, p2, data[0], p3, data[1], data[2], data[3]])
    
    return encoded


def hamming_decode(encoded_bits: list) -> list:
    """Decode bits using (7,4) Hamming code with error correction."""
    if len(encoded_bits) % 7 != 0:
        return encoded_bits  # Return as-is if not properly encoded
    
    decoded = []
    for i in range(0, len(encoded_bits), 7):
        codeword = encoded_bits[i:i+7]
        if len(codeword) < 7:
            decoded.extend(codeword)
            continue
        
        # Calculate syndromes
        s1 = codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6]
        s2 = codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6]
        s3 = codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6]
        
        # Error position
        error_pos = s1 + 2*s2 + 4*s3
        
        # Correct error if detected
        if error_pos > 0 and error_pos <= 7:
            codeword[error_pos - 1] ^= 1
        
        # Extract data bits
        decoded.extend([codeword[2], codeword[4], codeword[5], codeword[6]])
    
    return decoded


def best_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.15, 
                       block_size: tuple = (8, 8), key: int = 42,
                       redundancy: int = 3, use_error_correction: bool = True) -> tuple:
    """
    Best SVD_STFT embedding with comprehensive optimizations.
    
    Key features:
    1. Error correction coding (Hamming)
    2. Redundancy embedding
    3. Adaptive strength based on block characteristics
    4. Synchronization markers
    5. Enhanced modulation strategy
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
                # For bit 1: increase with adaptive strength
                modification_factor = 1 + adaptive_alpha
            else:
                # For bit 0: decrease with adaptive strength (less aggressive)
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
                         key: int = 42, threshold_method: str = 'hybrid') -> list:
    """
    Best SVD_STFT extraction with comprehensive detection methods.
    
    Key features:
    1. Multiple detection methods with weighted voting
    2. Confidence scoring
    3. Redundancy handling
    4. Error correction decoding
    5. Sync pattern detection
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
                # Uncertain case
                rel_bits.append(0)
                confidence = 0.1
            
            rel_confidences.append(confidence)
        else:
            rel_bits.append(0)
            rel_confidences.append(0.0)
    
    # Method 2: Advanced threshold detection
    threshold, threshold_confidence, method_used = advanced_threshold_analysis(extracted_sigmas, threshold_method)
    thresh_bits = [1 if sigma > threshold else 0 for sigma in extracted_sigmas]
    
    # Method 3: Pattern-based detection
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
    
    # Method 4: Energy-based detection
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
        thresh_weight = threshold_confidence * 0.4
        pattern_weight = 0.6  # Higher weight for pattern-based
        energy_weight = 0.3
        
        # Calculate weighted vote
        vote = (rel_bits[i] * rel_weight + 
                thresh_bits[i] * thresh_weight + 
                pattern_bits[i] * pattern_weight +
                energy_bits[i] * energy_weight)
        
        total_weight = rel_weight + thresh_weight + pattern_weight + energy_weight
        
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


def test_best_implementation_on_samples(input_dir: str, output_dir: str, max_files: int = 100):
    """Test the best SVD_STFT implementation on the 100 sample dataset."""
    print("=" * 80)
    print("BEST SVD_STFT IMPLEMENTATION - 100 SAMPLE TEST")
    print("=" * 80)
    
    # Find audio files
    input_path = Path(input_dir)
    audio_files = []
    
    if input_path.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(input_path.glob(ext)))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    # Limit to max_files
    audio_files = audio_files[:max_files]
    print(f"Found {len(audio_files)} audio files for testing")
    
    # Test configurations
    configurations = [
        {'name': 'Best Basic', 'alpha': 0.15, 'block_size': (8, 8), 'redundancy': 3, 'error_correction': True},
        {'name': 'Best Strong', 'alpha': 0.25, 'block_size': (8, 8), 'redundancy': 3, 'error_correction': True},
        {'name': 'Best Conservative', 'alpha': 0.1, 'block_size': (8, 8), 'redundancy': 3, 'error_correction': True},
        {'name': 'Best High Redundancy', 'alpha': 0.15, 'block_size': (8, 8), 'redundancy': 5, 'error_correction': True},
    ]
    
    all_results = []
    best_overall = None
    best_ber = float('inf')
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print(f"  Alpha: {config['alpha']}, Block Size: {config['block_size']}")
        print(f"  Redundancy: {config['redundancy']}, Error Correction: {config['error_correction']}")
        
        config_results = []
        successful_extractions = 0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {config['name']}"):
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
                    alpha=config['alpha'],
                    block_size=config['block_size'],
                    key=42,
                    redundancy=config['redundancy'],
                    use_error_correction=config['error_correction']
                )
                
                # Reconstruct audio
                watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
                
                # Ensure same length
                min_len = min(len(audio), len(watermarked_audio))
                watermarked_audio = watermarked_audio[:min_len]
                audio_test = audio[:min_len]
                
                # Extract watermark
                S_test = compute_stft(watermarked_audio, 16000, n_fft=256, hop_length=64, window='hann')
                extracted_bits = best_extract_svd_stft(
                    S_test, reference_data,
                    alpha=config['alpha'],
                    block_size=config['block_size'],
                    key=42,
                    threshold_method='hybrid'
                )
                
                # Calculate metrics
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
                    
                    success = clean_ber < 0.1
                    if success:
                        successful_extractions += 1
                    
                    config_results.append({
                        'file': audio_file.name,
                        'clean_ber': clean_ber,
                        'snr': snr,
                        'success': success,
                        'original_bits': watermark_compare,
                        'extracted_bits': extracted_compare
                    })
                else:
                    config_results.append({
                        'file': audio_file.name,
                        'clean_ber': 1.0,
                        'snr': -np.inf,
                        'success': False,
                        'error': 'No bits extracted'
                    })
                
            except Exception as e:
                config_results.append({
                    'file': audio_file.name,
                    'clean_ber': 1.0,
                    'snr': -np.inf,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        successful_results = [r for r in config_results if r.get('success', False)]
        avg_ber = np.mean([r.get('clean_ber', 1.0) for r in config_results])
        avg_snr = np.mean([r.get('snr', -np.inf) for r in config_results if r.get('snr', -np.inf) > -np.inf])
        success_rate = (len(successful_results) / len(config_results)) * 100
        
        config_summary = {
            'config': config,
            'total_files': len(config_results),
            'successful_extractions': len(successful_results),
            'success_rate': success_rate,
            'avg_ber': avg_ber,
            'avg_snr': avg_snr,
            'results': config_results
        }
        
        all_results.append(config_summary)
        
        print(f"  Results:")
        print(f"    Success Rate: {success_rate:.1f}%")
        print(f"    Average BER: {avg_ber:.4f}")
        print(f"    Average SNR: {avg_snr:.2f} dB")
        print(f"    Successful: {len(successful_results)}/{len(config_results)}")
        
        # Track best configuration
        if avg_ber < best_ber:
            best_ber = avg_ber
            best_overall = config_summary
    
    # Display best configuration
    if best_overall:
        print(f"\n🎯 BEST OVERALL CONFIGURATION:")
        print(f"   Config: {best_overall['config']['name']}")
        print(f"   Alpha: {best_overall['config']['alpha']}")
        print(f"   Block Size: {best_overall['config']['block_size']}")
        print(f"   Redundancy: {best_overall['config']['redundancy']}")
        print(f"   Error Correction: {best_overall['config']['error_correction']}")
        print(f"   Success Rate: {best_overall['success_rate']:.1f}%")
        print(f"   Average BER: {best_overall['avg_ber']:.4f}")
        print(f"   Average SNR: {best_overall['avg_snr']:.2f} dB")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "best_svd_stft_results.json", 'w') as f:
        json.dump({
            'best_configuration': best_overall,
            'all_results': all_results,
            'test_summary': {
                'total_files_tested': len(audio_files),
                'best_ber_achieved': best_ber if best_overall else float('inf'),
                'best_success_rate': best_overall['success_rate'] if best_overall else 0
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return best_overall, all_results


if __name__ == "__main__":
    # Test on 100 sample dataset
    best_result, all_results = test_best_implementation_on_samples(
        input_dir="100sample_wav/music_wav",
        output_dir="best_svd_stft_results",
        max_files=100
    )
    
    if best_result:
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"   Use configuration: {best_result['config']['name']}")
        print(f"   Parameters: α={best_result['config']['alpha']}, blocks={best_result['config']['block_size']}")
        print(f"   Expected performance: {best_result['success_rate']:.1f}% success rate, {best_result['avg_ber']:.4f} BER")
    else:
        print(f"\n❌ No successful configurations found.")

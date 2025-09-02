#!/usr/bin/env python3
"""
Improved SVD_STFT watermarking algorithm with enhanced threshold techniques.
Addresses fundamental extraction issues and provides better robustness.
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


def adaptive_threshold(sigmas: list, method: str = 'otsu') -> float:
    """
    Calculate adaptive threshold using various methods.
    
    Args:
        sigmas: List of singular values
        method: Threshold method ('otsu', 'kmeans', 'percentile', 'entropy')
    
    Returns:
        Optimal threshold value
    """
    if not sigmas:
        return 0.0
    
    sigmas = np.array(sigmas)
    
    if method == 'otsu':
        # Otsu's method for thresholding
        hist, bins = np.histogram(sigmas, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate Otsu threshold
        total_pixels = len(sigmas)
        total_mean = np.mean(sigmas)
        
        best_threshold = 0
        best_variance = 0
        
        for i, threshold in enumerate(bin_centers):
            # Split into two classes
            class1 = sigmas[sigmas <= threshold]
            class2 = sigmas[sigmas > threshold]
            
            if len(class1) == 0 or len(class2) == 0:
                continue
            
            # Calculate class probabilities
            p1 = len(class1) / total_pixels
            p2 = len(class2) / total_pixels
            
            # Calculate class means
            mean1 = np.mean(class1)
            mean2 = np.mean(class2)
            
            # Calculate between-class variance
            variance = p1 * p2 * (mean1 - mean2) ** 2
            
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
        
        return best_threshold
    
    elif method == 'kmeans':
        # Simple k-means clustering
        from sklearn.cluster import KMeans
        
        if len(sigmas) < 2:
            return np.mean(sigmas)
        
        # Reshape for sklearn
        X = sigmas.reshape(-1, 1)
        
        # Apply k-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Return the threshold as the midpoint between cluster centers
        centers = sorted(kmeans.cluster_centers_.flatten())
        return (centers[0] + centers[1]) / 2
    
    elif method == 'percentile':
        # Percentile-based threshold
        return np.percentile(sigmas, 75)  # 75th percentile
    
    elif method == 'entropy':
        # Entropy-based thresholding
        hist, bins = np.histogram(sigmas, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        best_threshold = 0
        best_entropy = 0
        
        for threshold in bin_centers:
            # Split into two classes
            class1 = sigmas[sigmas <= threshold]
            class2 = sigmas[sigmas > threshold]
            
            if len(class1) == 0 or len(class2) == 0:
                continue
            
            # Calculate entropy for each class
            hist1, _ = np.histogram(class1, bins=25)
            hist2, _ = np.histogram(class2, bins=25)
            
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-10)
            hist2 = hist2 / (np.sum(hist2) + 1e-10)
            
            # Calculate entropy
            entropy1 = entropy(hist1 + 1e-10)
            entropy2 = entropy(hist2 + 1e-10)
            
            # Combined entropy
            total_entropy = entropy1 + entropy2
            
            if total_entropy > best_entropy:
                best_entropy = total_entropy
                best_threshold = threshold
        
        return best_threshold
    
    else:
        # Default to median
        return np.median(sigmas)


def enhanced_embed_svd_stft(S_complex: np.ndarray, bits: list, alpha: float = 0.1, 
                           block_size: tuple = (8, 8), key: int = 42,
                           redundancy: int = 3) -> tuple:
    """
    Enhanced SVD_STFT embedding with redundancy and better modulation.
    
    Args:
        S_complex: Complex STFT matrix
        bits: Watermark bits
        alpha: Watermark strength
        block_size: Block dimensions
        key: Random seed
        redundancy: Number of redundant embeddings per bit
    
    Returns:
        Tuple of (modified_STFT, reference_data)
    """
    from utils.blocks import split_into_blocks, reassemble_blocks, pseudo_permutation
    
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Split into blocks
    blocks = split_into_blocks(S_mag, block_size)
    n_blocks = len(blocks)
    
    if n_blocks < len(bits) * redundancy:
        raise ValueError(f"Not enough blocks ({n_blocks}) for {len(bits)} bits with redundancy {redundancy}")
    
    # Generate permutation
    perm = pseudo_permutation(n_blocks, key)
    
    # Store reference data
    reference_data = {
        'original_sigmas': [],
        'modified_sigmas': [],
        'bit_assignments': [],
        'block_indices': []
    }
    
    # Embed watermark with redundancy
    bit_idx = 0
    for i, bit in enumerate(bits):
        # Embed the same bit in multiple blocks for redundancy
        for r in range(redundancy):
            if bit_idx >= len(perm):
                break
            
            block_idx = perm[bit_idx]
            block = blocks[block_idx]
            
            # Compute SVD
            U, Sigma, Vt = np.linalg.svd(block, full_matrices=False)
            
            # Store original singular value
            original_sigma = float(Sigma[0])
            reference_data['original_sigmas'].append(original_sigma)
            reference_data['bit_assignments'].append(bit)
            reference_data['block_indices'].append(block_idx)
            
            # Enhanced modification based on bit value
            if bit == 1:
                # For bit 1: increase the singular value with adaptive strength
                modification_factor = 1 + alpha * (1 + 0.1 * np.random.random())
                Sigma[0] *= modification_factor
            else:
                # For bit 0: decrease the singular value with adaptive strength
                modification_factor = 1 - alpha * (1 + 0.1 * np.random.random())
                Sigma[0] *= max(modification_factor, 0.1)  # Ensure positive
            
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


def enhanced_extract_svd_stft(S_complex_mod: np.ndarray, reference_data: dict,
                             alpha: float = 0.1, block_size: tuple = (8, 8), 
                             key: int = 42, num_bits: int = None,
                             threshold_method: str = 'otsu') -> list:
    """
    Enhanced SVD_STFT extraction with multiple threshold techniques.
    
    Args:
        S_complex_mod: Modified complex STFT matrix
        reference_data: Reference data from embedding
        alpha: Watermark strength
        block_size: Block dimensions
        key: Random seed
        num_bits: Number of bits to extract
        threshold_method: Threshold method to use
    
    Returns:
        Extracted watermark bits
    """
    from utils.blocks import split_into_blocks, pseudo_permutation
    
    S_mag_mod = np.abs(S_complex_mod)
    
    # Split into blocks
    blocks_mod = split_into_blocks(S_mag_mod, block_size)
    n_blocks = len(blocks_mod)
    
    if num_bits is None:
        num_bits = len(reference_data['bit_assignments']) // 3  # Assuming redundancy=3
    
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
    
    # Method 1: Reference-based extraction
    ref_bits = []
    if len(reference_data['original_sigmas']) == len(extracted_sigmas):
        for i, (orig_sigma, ext_sigma) in enumerate(zip(reference_data['original_sigmas'], extracted_sigmas)):
            # Calculate expected values
            expected_high = orig_sigma * (1 + alpha)
            expected_low = orig_sigma * (1 - alpha)
            
            # Determine bit based on distance to expected values
            dist_to_high = abs(ext_sigma - expected_high)
            dist_to_low = abs(ext_sigma - expected_low)
            
            if dist_to_high < dist_to_low:
                ref_bits.append(1)
            else:
                ref_bits.append(0)
    
    # Method 2: Adaptive threshold extraction
    threshold = adaptive_threshold(extracted_sigmas, threshold_method)
    thresh_bits = [1 if sigma > threshold else 0 for sigma in extracted_sigmas]
    
    # Method 3: Relative change extraction
    rel_bits = []
    for i, (orig_sigma, ext_sigma) in enumerate(zip(reference_data['original_sigmas'], extracted_sigmas)):
        if orig_sigma > 0:
            relative_change = (ext_sigma - orig_sigma) / orig_sigma
            if relative_change > alpha * 0.5:  # Threshold at half alpha
                rel_bits.append(1)
            else:
                rel_bits.append(0)
        else:
            rel_bits.append(0)
    
    # Combine methods using voting
    final_bits = []
    redundancy = len(extracted_sigmas) // num_bits
    
    for i in range(num_bits):
        start_idx = i * redundancy
        end_idx = start_idx + redundancy
        
        if end_idx <= len(ref_bits):
            # Get bits from all methods for this watermark bit
            ref_votes = ref_bits[start_idx:end_idx]
            thresh_votes = thresh_bits[start_idx:end_idx]
            rel_votes = rel_bits[start_idx:end_idx]
            
            # Count votes for each method
            ref_count = sum(ref_votes)
            thresh_count = sum(thresh_votes)
            rel_count = sum(rel_votes)
            
            # Majority voting
            total_votes = ref_count + thresh_count + rel_count
            if total_votes > (len(ref_votes) + len(thresh_votes) + len(rel_votes)) / 2:
                final_bits.append(1)
            else:
                final_bits.append(0)
        else:
            final_bits.append(0)
    
    return final_bits


def test_improved_algorithm():
    """Test the improved SVD_STFT algorithm."""
    print("=" * 80)
    print("IMPROVED SVD_STFT ALGORITHM TESTING")
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
        {'name': 'Enhanced Basic', 'alpha': 0.1, 'block_size': (8, 8), 'redundancy': 3},
        {'name': 'Enhanced Strong', 'alpha': 0.2, 'block_size': (8, 8), 'redundancy': 3},
        {'name': 'Enhanced Large Blocks', 'alpha': 0.1, 'block_size': (16, 16), 'redundancy': 3},
        {'name': 'Enhanced High Redundancy', 'alpha': 0.1, 'block_size': (8, 8), 'redundancy': 5},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        # Embed watermark
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, reference_data = enhanced_embed_svd_stft(
            S, watermark_bits, 
            alpha=config['alpha'], 
            block_size=config['block_size'], 
            key=42,
            redundancy=config['redundancy']
        )
        watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
        
        # Ensure same length
        min_len = min(len(audio), len(watermarked_audio))
        watermarked_audio = watermarked_audio[:min_len]
        audio_test = audio[:min_len]
        
        # Test different threshold methods
        threshold_methods = ['otsu', 'kmeans', 'percentile', 'entropy']
        
        for method in threshold_methods:
            try:
                # Test clean extraction
                S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
                extracted_bits = enhanced_extract_svd_stft(
                    S_test, reference_data, 
                    alpha=config['alpha'], 
                    block_size=config['block_size'], 
                    key=42, 
                    num_bits=len(watermark_bits),
                    threshold_method=method
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
                    'config': config['name'],
                    'threshold_method': method,
                    'alpha': config['alpha'],
                    'block_size': config['block_size'],
                    'redundancy': config['redundancy'],
                    'clean_ber': clean_ber,
                    'snr': snr,
                    'success': clean_ber < 0.1,
                    'extracted_bits': extracted_bits
                })
                
                print(f"  {method}: BER={clean_ber:.4f}, SNR={snr:.2f} dB, Success={'✓' if clean_ber < 0.1 else '✗'}")
                
            except Exception as e:
                print(f"  {method}: Error - {e}")
                results.append({
                    'config': config['name'],
                    'threshold_method': method,
                    'alpha': config['alpha'],
                    'block_size': config['block_size'],
                    'redundancy': config['redundancy'],
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
        print(f"   Threshold Method: {best_result['threshold_method']}")
        print(f"   Alpha: {best_result['alpha']}")
        print(f"   Block Size: {best_result['block_size']}")
        print(f"   Redundancy: {best_result['redundancy']}")
        print(f"   Clean BER: {best_result['clean_ber']:.4f}")
        print(f"   SNR: {best_result['snr']:.2f} dB")
        
        # Test robustness for best configuration
        print(f"\nTesting robustness for best configuration...")
        
        # Re-embed with best parameters
        S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
        S_watermarked, reference_data = enhanced_embed_svd_stft(
            S, watermark_bits, 
            alpha=best_result['alpha'], 
            block_size=best_result['block_size'], 
            key=42,
            redundancy=best_result['redundancy']
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
            return enhanced_extract_svd_stft(
                S, reference_data,
                params['alpha'],
                params['block_size'],
                params['key'],
                params['num_bits'],
                params['threshold_method']
            )
        
        extract_params = {
            'alpha': best_result['alpha'],
            'block_size': best_result['block_size'],
            'key': 42,
            'num_bits': len(watermark_bits),
            'threshold_method': best_result['threshold_method']
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
        output_dir = Path("improved_algorithm_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save audio files
        sf.write(output_dir / "original.wav", audio, sample_rate)
        sf.write(output_dir / "watermarked.wav", watermarked_audio, sample_rate)
        
        # Save detailed results
        with open(output_dir / "improved_results.json", 'w') as f:
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
    best_result, attack_results = test_improved_algorithm()
    
    if best_result:
        print(f"\n🎯 RECOMMENDED IMPROVED PARAMETERS:")
        print(f"   Config: {best_result['config']}")
        print(f"   Threshold Method: {best_result['threshold_method']}")
        print(f"   Alpha: {best_result['alpha']}")
        print(f"   Block Size: {best_result['block_size']}")
        print(f"   Redundancy: {best_result['redundancy']}")
        print(f"   Expected Clean BER: {best_result['clean_ber']:.4f}")
        if attack_results:
            success_rate = sum(1 for r in attack_results.values() if r.get('success', False)) / len(attack_results) * 100
            print(f"   Expected Attack Success Rate: {success_rate:.1f}%")
    else:
        print(f"\n❌ No optimal parameters found. Consider further algorithm improvements.")

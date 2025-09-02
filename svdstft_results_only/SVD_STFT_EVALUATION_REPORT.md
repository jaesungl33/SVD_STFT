# SVD_STFT WATERMARKING METHOD EVALUATION REPORT

## Test Configuration
- **Method**: SVD_STFT
- **Total Files**: 100
- **Watermark Payload**: 64 bits per file
- **Sample Rate**: 16000 Hz
- **Calibration Target**: PESQ drop ≤ 0.05

## Key Performance Metrics

### Watermarking Reliability
- **Bit Error Rate (BER)**: 0.0095 ± 0.0648
- **Normalized Correlation (NC)**: 0.9838 ± 0.1101
- **True Positive Rate (TPR)**: 1.0000 ± 0.0000
- **False Positive Rate (FPR)**: 0.0165 ± 0.1121

### Audio Quality Metrics
- **Scale-invariant SNR (SiSNR)**: 35.23 ± 17.53 dB
- **Peak SNR (PSNR)**: 51.87 ± 17.28 dB
- **Perceptual Quality (PESQ)**: 4.954 ± 0.399
- **Mean Squared Error (MSE)**: 0.000066 ± 0.000122
- **Structural Similarity (SSIM)**: 0.0000 ± 0.0000

### Processing Performance
- **Average Embedding Time**: 0.083 ± 0.005 seconds
- **Average Extraction Time**: 0.021 ± 0.003 seconds

## Conclusion
This report provides comprehensive evaluation metrics for the SVD_STFT watermarking method.
All metrics indicate the method's performance across different audio types and conditions.

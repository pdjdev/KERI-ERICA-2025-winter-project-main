import numpy as np
from scipy.stats import entropy

def fft_features(daily_seq, top_k=3):
    """주어진 시퀀스(28일)에 대해 FFT 특징 추출"""
    n = len(daily_seq) # n = 28
    detrended_seq = daily_seq - np.mean(daily_seq) # daily_seq에서 평균 뺀 행렬(DC Offset 제거)
    fft_vals = np.fft.rfft(detrended_seq) # Discreate Fourier Transform
    amplitudes = np.abs(fft_vals) # 복소평면 속 유클리디안 거리 -> 진폭 계산
    freqs = np.fft.rfftfreq(n) # 주파수 행렬 생성
    
    features = {}
    
    # 상위 K개 진폭 및 해당 주파수
    top_indices = np.argsort(amplitudes)[-top_k:][::-1] # 상위 진폭 3개 추출
    for j, idx in enumerate(top_indices): # j=0,1,2; idx=진폭 인덱스 6개 Feature 생성
        features[f'fft_amp_{j}'] = amplitudes[idx]
        features[f'fft_freq_{j}'] = freqs[idx]
        
    # Spectral
    psd = amplitudes**2 # Power Spectral Density, 28일 주기, 14일 주기, ... 2일 주기, 1일 주기 신호의 에너지 분포
    features['total_power'] = np.sum(psd) # 전체 에너지 합
    
    # Spectral Entropy
    psd_norm = psd / (np.sum(psd) + 1e-9) # 확률분포 정규화
    features['spectral_entropy'] = entropy(psd_norm) # 엔트로피 계산
    
    # Band Power
    mid_idx = len(psd) // 2
    features['low_freq_power'] = np.sum(psd[:mid_idx]) # 저주파 에너지 합
    features['high_freq_power'] = np.sum(psd[mid_idx:]) # 고주파 에너지 합
    
    return features
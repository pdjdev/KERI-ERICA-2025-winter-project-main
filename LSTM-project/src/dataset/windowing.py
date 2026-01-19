import pandas as pd
import numpy as np

from src.features.fourier_transform import fft_features

def build_sliding_window_samples(type_df, lookback=28, horizon=7, top_k=3):
    """특정 충전방식 데이터에 대해 슬라이딩 윈도우 샘플 생성"""
    type_df = type_df.sort_values('일자').reset_index(drop=True)
    n_rows = len(type_df)
    samples = []
    
    # i는 현재(t) 시점의 인덱스
    for i in range(lookback - 1, n_rows - horizon):
        # 1. 시퀀스 데이터 추출 (Leakage 방지: t+1 이후 데이터 미포함)
        window = type_df.iloc[i - (lookback-1) : i + 1]
        target_window = type_df.iloc[i + 1 : i + 1 + horizon]
        
        daily_seq = window['daily_total'].values
        
        # 2. 특징 구성 (Time-domain)
        current_date = type_df.iloc[i]['일자']
        feat = {
            'window_end_date': current_date,
            'charging_type': type_df.iloc[i]['충전방식'],
            'y_next7_total': target_window['daily_total'].sum(),
            'month': current_date.month,
            'dayofweek': current_date.dayofweek,
            'mean_28d': np.mean(daily_seq),
            'std_28d': np.std(daily_seq),
            'last_day_usage': daily_seq[-1],
            'peak_ratio_mean': window['peak_ratio'].mean()
        }
        
        # 3. FFT 특징 결합
        fft_feat = fft_features(daily_seq, top_k=top_k)
        feat.update(fft_feat)
        
        samples.append(feat)
        
    return samples

def build_train_df(df, lookback=28, horizon=7, top_k=3):
    """전체 데이터에 대해 충전방식별 윈도우를 생성하고 병합"""
    all_samples = []
    charging_types = df['충전방식'].unique()
    
    for c_type in charging_types:
        type_df = df[df['충전방식'] == c_type]
        type_samples = build_sliding_window_samples(type_df, lookback, horizon, top_k)
        all_samples.extend(type_samples)
        print(f"Windows generated for {c_type}: {len(type_samples)}")
        
    train_df = pd.DataFrame(all_samples)
    
    # 시간 순 정렬 (중요: 검증 시 Leakage 방지)
    train_df = train_df.sort_values(['window_end_date', 'charging_type']).reset_index(drop=True)
    return train_df
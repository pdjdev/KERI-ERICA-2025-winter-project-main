import pandas as pd

def preprocess_data(data):
    """데이터 기본 정제 및 타입 변환"""
    df = data.copy()
    df['일자'] = pd.to_datetime(df['일자'])
    
    # 0시~23시 컬럼 수치형 변환 및 결측치 처리
    hourly_cols = [f'{i}시' for i in range(24)]
    for col in hourly_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def add_daily_features(df):
    """일 단위 집계 및 시간대별 특성 추가"""
    hourly_cols = [f'{i}시' for i in range(24)]
    
    # 일일 총 사용량
    df['daily_total'] = df[hourly_cols].sum(axis=1)
    
    # Peak 시간대(08-20시) 비중 계산
    peak_cols = [f'{i}시' for i in range(8, 20)]
    df['peak_ratio'] = df[peak_cols].sum(axis=1) / (df['daily_total'] + 1e-9)
    
    return df
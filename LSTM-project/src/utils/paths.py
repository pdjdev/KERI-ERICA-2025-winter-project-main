"""
프로젝트 경로 관리 유틸리티
"""
from pathlib import Path

# 프로젝트 루트 디렉토리 (src의 부모의 부모)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 원본 데이터 디렉토리
ORIGINAL_DATA_DIR = PROJECT_ROOT / "original_Data"

# 기본 데이터셋 파일
DATASET_PATH = INTERIM_DATA_DIR / "dataset.csv"

# 소스 코드 디렉토리
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = SRC_DIR / "config"

# 출력 디렉토리
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# 노트북 디렉토리
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# 스크립트 디렉토리
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_dir(path: Path) -> Path:
    """디렉토리가 없으면 생성"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_path(filename: str) -> Path:
    """원본 데이터 파일 경로 반환"""
    return RAW_DATA_DIR / filename


def get_interim_data_path(filename: str) -> Path:
    """중간 데이터 파일 경로 반환"""
    return INTERIM_DATA_DIR / filename


def get_processed_data_path(filename: str) -> Path:
    """최종 처리 데이터 파일 경로 반환"""
    return PROCESSED_DATA_DIR / filename


def get_model_path(filename: str) -> Path:
    """모델 저장 경로 반환"""
    ensure_dir(MODELS_DIR)
    return MODELS_DIR / filename


def get_figure_path(filename: str) -> Path:
    """그래프 저장 경로 반환"""
    ensure_dir(FIGURES_DIR)
    return FIGURES_DIR / filename


def get_report_path(filename: str) -> Path:
    """리포트 저장 경로 반환"""
    ensure_dir(REPORTS_DIR)
    return REPORTS_DIR / filename


if __name__ == "__main__":
    # 경로 확인용
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"INTERIM_DATA_DIR: {INTERIM_DATA_DIR}")
    print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"FIGURES_DIR: {FIGURES_DIR}")
    print(f"REPORTS_DIR: {REPORTS_DIR}")

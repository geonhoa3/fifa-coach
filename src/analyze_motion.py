"""
Step 4a — 공 속도 계산 + 행동 1차 분류

흐름:
  1. game_state.csv 로딩
  2. 연속 검출 프레임 사이 속도(px/frame) 계산
  3. 임계값으로 hold / dribble / pass_or_shot 분류
  4. 분포 통계 + 텍스트 히스토그램 출력
  5. motion.csv 저장

실행:
  python -m src.analyze_motion
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


CSV_IN = "data/output/game_state.csv"
META_IN = "data/output/game_state.meta.json"
CSV_OUT = "data/output/motion.csv"


def compute_speed(csv_path: str = CSV_IN, meta_path: str = META_IN) -> pd.DataFrame:
    """공 위치 시계열 → 속도 컬럼 추가."""
    df = pd.read_csv(csv_path)
    meta = json.loads(Path(meta_path).read_text())
    sample_every = meta["sample_every"]

    df["dx"] = df["ball_x"].diff()
    df["dy"] = df["ball_y"].diff()
    df["dframe"] = df["frame"].diff()
    dist = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["speed_px_per_frame"] = dist / df["dframe"]

    # dframe 이 sample_every 와 정확히 일치할 때만 유효 (재등장 점프 제거)
    bad = (
        (df["dframe"] != sample_every)
        | df["ball_x"].isna()
        | df["ball_y"].isna()
        | df["ball_x"].shift().isna()
        | df["ball_y"].shift().isna()
    )
    df.loc[bad, "speed_px_per_frame"] = np.nan
    return df


def classify_action(df: pd.DataFrame, slow: float = 2.0, fast: float = 8.0) -> pd.DataFrame:
    """속도 → 행동 라벨."""
    df["action"] = "no_ball"
    s = df["speed_px_per_frame"]
    df.loc[s.notna() & (s < slow), "action"] = "hold"
    df.loc[s.notna() & (s >= slow) & (s < fast), "action"] = "dribble"
    df.loc[s.notna() & (s >= fast), "action"] = "pass_or_shot"
    return df


def print_text_histogram(values: pd.Series, bins: int = 25, max_speed: float = 10):
    """터미널에 히스토그램 한 줄씩 출력."""
    s = values.dropna().clip(0, max_speed)
    if len(s) == 0:
        print("  (no data)")
        return
    # 콘솔 인코딩(cp949 등)이 '█' 를 못 그리면 '#' 로 대체 (crash 방지)
    import sys
    bar_char = "█"
    try:
        bar_char.encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        bar_char = "#"
    counts, edges = np.histogram(s, bins=bins, range=(0, max_speed))
    cap = max(counts.max(), 1)
    print(f"  speed(px/frame)  |  count")
    for i in range(bins):
        bar = bar_char * int(counts[i] * 40 / cap)
        print(f"  {edges[i]:5.1f} - {edges[i+1]:5.1f}  | {bar} ({counts[i]})")


def summarize(df: pd.DataFrame, slow: float, fast: float):
    s = df["speed_px_per_frame"]
    valid = s.dropna()
    print(f"\n[속도 통계] 유효 샘플 {len(valid)}")
    if len(valid):
        print(f"  평균    : {valid.mean():.2f}")
        print(f"  중앙값  : {valid.median():.2f}")
        print(f"  75%분위 : {valid.quantile(0.75):.2f}")
        print(f"  90%분위 : {valid.quantile(0.90):.2f}")
        print(f"  95%분위 : {valid.quantile(0.95):.2f}")
        print(f"  99%분위 : {valid.quantile(0.99):.2f}")
        print(f"  최댓값  : {valid.max():.2f}")

    print(f"\n[히스토그램]  slow={slow}  fast={fast}")
    print_text_histogram(valid)

    print(f"\n[행동 분류]")
    print(df["action"].value_counts().to_string())


if __name__ == "__main__":
    SLOW, FAST = 0.8, 2.0
    df = compute_speed()
    df = classify_action(df, slow=SLOW, fast=FAST)
    summarize(df, slow=SLOW, fast=FAST)

    Path(CSV_OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n[저장] {CSV_OUT}")

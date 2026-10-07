"""
Step 5a — 그리드 정의 + 공격 방향 정규화

미니맵 (276x156) → GRID_W x GRID_H 셀.
ball_x, ball_y → (i, j) 변환.

우리팀 공격 방향 자동 추정 후 정규화:
  - 우리팀 dot 평균 x 가 미니맵 좌측 절반 → 우측 공격 (+1)
  - 우측 절반 → 좌측 공격 (-1)
  - direction=-1 이면 i 를 뒤집어서 "큰 i = 상대 골대 방향" 통일

관전 모드 (events.csv 에 team 컬럼 있음):
  - 양팀 attack direction 동시 추정 (paired)
  - 각 이벤트의 team 에 따라 i 정규화

실행:
  python -m src.grid
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


GRID_W = 16
GRID_H = 12

DEFAULT_CSV_IN = "data/output/events.csv"
DEFAULT_CSV_OUT = "data/output/events.csv"
DEFAULT_META_OUT = "data/output/grid_meta.json"


def load_minimap_size(video_path: str = None) -> tuple:
    """미니맵 ROI 의 (w, h) 반환. 영상별 ROI 구조 + 옛 단일 구조 지원."""
    from src.minimap import load_roi
    roi = load_roi(video_path=video_path)
    return roi[2], roi[3]


def detect_attack_direction(df: pd.DataFrame, our_team: int = 0,
                             mini_w: int = 276) -> int:
    """단일 팀 공격 방향 추정 (그 팀의 dot 평균 x 기준)."""
    col = "t0_dots" if our_team == 0 else "t1_dots"
    xs = []
    for s in df[col].dropna():
        if isinstance(s, str) and s and s != "[]":
            try:
                xs.extend([d[0] for d in json.loads(s)])
            except Exception:
                pass
    if not xs:
        return 1
    avg_x = float(np.mean(xs))
    print(f"[방향 추정] T{our_team} dot 평균 x = {avg_x:.1f} (미니맵 폭 {mini_w})")
    return 1 if avg_x < mini_w / 2 else -1


def detect_team_directions_paired(df: pd.DataFrame,
                                    mini_w: int = 276) -> tuple:
    """양팀 방향 동시 추정. 두 팀 dot 평균 x 비교해서 반대 방향 강제.

    좌측 평균인 팀이 우측 공격 (+1), 우측 평균인 팀이 좌측 공격 (-1).
    한 팀씩 따로 보면 양팀이 같은 쪽으로 판정될 수 있어 paired 가 견고.
    """
    def avg_x(col):
        xs = []
        for s in df[col].dropna():
            if isinstance(s, str) and s and s != "[]":
                try:
                    xs.extend([d[0] for d in json.loads(s)])
                except Exception:
                    pass
        return float(np.mean(xs)) if xs else mini_w / 2

    x0 = avg_x("t0_dots")
    x1 = avg_x("t1_dots")
    print(f"[방향 추정 (paired)] T0 평균 x = {x0:.1f}, T1 평균 x = {x1:.1f}")
    if x0 < x1:
        return 1, -1
    return -1, 1


def estimate_direction_series(df: pd.DataFrame, mini_w: int) -> np.ndarray:
    """프레임별 T0 공격 방향 시계열 (+1/-1). 골키퍼 신호 기반.

    기존 '팀 평균 x' 방식의 결함 두 가지를 해결:
      1. 평균 x 차이는 수 px (노이즈 수준) → 최후방 dot(골키퍼)은
         자기 골대에 붙어 있어 80px+ 차이로 명확
      2. 영상 전체 단일 방향 → 하프타임 진영 교대 무시.

    교대는 '하프타임 1회'로 제약: 프레임 신호와 가장 많이 일치하는
    (전반 부호, 교대 시점) 하나를 고른다. 영상 안에 교대가 없으면
    교대 시점이 끝에 붙어 단일 방향이 된다.
    이전엔 rolling median(151행)이라 경기당 3~11회 가짜 교대가 생겼고,
    골 위치 vs 스코어보드 대조에서 골 123개 중 88개(72%)만 일치했다
    (match19는 경기 전체 반전). 1회 제약 후 117개(95%) 일치.

    반환: len(df) 크기의 +1/-1 배열 (T0 기준. T1 은 항상 반대)
    """
    sig = np.full(len(df), np.nan)
    for k, (_, r) in enumerate(df.iterrows()):
        try:
            x0 = [d[0] for d in json.loads(r["t0_dots"])]
            x1 = [d[0] for d in json.loads(r["t1_dots"])]
        except (TypeError, ValueError):
            continue
        if len(x0) < 5 or len(x1) < 5:
            continue
        # T0 왼쪽 수비 가정 잔차 vs 오른쪽 수비 가정 잔차 (작을수록 그럴듯)
        left = min(x0) + (mini_w - max(x1))
        right = (mini_w - max(x0)) + min(x1)
        sig[k] = 1.0 if left < right else -1.0
    if np.isnan(sig).all():
        return np.ones(len(df), dtype=int)
    # t 까지 부호 s, 이후 -s 일 때 일치 점수 = s*(앞 합) - s*(뒤 합)
    c = np.cumsum(np.nan_to_num(sig))
    tot = c[-1]
    score = c - (tot - c)                     # s=+1 기준; s=-1 은 부호 반대
    t_pos, t_neg = int(np.argmax(score)), int(np.argmin(score))
    if score[t_pos] >= -score[t_neg]:
        t, s = t_pos, 1
    else:
        t, s = t_neg, -1
    # 교대를 넣어 얻는 일치 증가가 작으면 교대 없음 (하프타임 전 끝난 짧은 영상).
    # 실측: 정상 경기 37~98%, 짧은 영상(match08·12·16·20) 0~3%.
    n_sig = int((~np.isnan(sig)).sum())
    if (score[t] * s - abs(tot)) < 0.15 * n_sig:
        t, s = len(df) - 1, (1 if tot >= 0 else -1)
    dirs = np.where(np.arange(len(df)) <= t, s, -s)
    has = ~np.isnan(sig)
    agree = (sig[has] == dirs[has]).mean()
    swap = t < len(df) - 1
    print(f"[방향 시계열] 전반 T0 {s:+d}, "
          + (f"교대 {df['time_sec'].iloc[t] / 60:.1f}분" if swap and "time_sec" in df
             else "교대 없음")
          + f", 프레임 신호 일치 {agree * 100:.0f}%")
    return dirs


def add_grid_columns(df: pd.DataFrame, direction,
                      mini_w: int, mini_h: int) -> pd.DataFrame:
    """ball_x,y → 그리드 i,j (원본 + 정규화) 컬럼 추가.

    direction: int(+1/-1, 영상 전체 단일) 또는 프레임별 배열.
    """
    raw_i = (df["ball_x"] / mini_w * GRID_W).clip(0, GRID_W - 1)
    raw_j = (df["ball_y"] / mini_h * GRID_H).clip(0, GRID_H - 1)
    df["ball_i"] = np.floor(raw_i)
    df["ball_j"] = np.floor(raw_j)
    d = np.asarray(direction) if not np.isscalar(direction) else np.full(len(df), direction)
    df["ball_i_norm"] = np.where(d == 1, df["ball_i"], (GRID_W - 1) - df["ball_i"])
    df.loc[df["ball_i"].isna(), "ball_i_norm"] = np.nan
    df["ball_j_norm"] = df["ball_j"]
    return df


def add_grid_columns_observer(df: pd.DataFrame, dir_t0_series,
                                mini_w: int, mini_h: int) -> pd.DataFrame:
    """관전 모드: 팀별 attack direction 으로 i_norm 따로 정규화.

    dir_t0_series: 프레임별 T0 방향 (+1/-1) 배열. T1 은 항상 반대.
    """
    raw_i = (df["ball_x"] / mini_w * GRID_W).clip(0, GRID_W - 1)
    raw_j = (df["ball_y"] / mini_h * GRID_H).clip(0, GRID_H - 1)
    df["ball_i"] = np.floor(raw_i)
    df["ball_j"] = np.floor(raw_j)
    df["ball_j_norm"] = df["ball_j"]
    d0 = np.asarray(dir_t0_series)

    def _norm_i(pos, row_k):
        if pd.isna(pos):
            return np.nan
        row = df.iloc[row_k]
        # 정규화 우선순위: team (이벤트 행위자) > pos_team (현재 점유).
        # 슛 순간엔 공이 골문에 있어 pos_team 이 수비 GK 팀으로 잡힘 →
        # 행위자 기준이어야 슛이 상대 골대(i≈15)로 정규화됨.
        for key in ("team", "pos_team"):
            v = row.get(key)
            if v is not None and not pd.isna(v):
                d = d0[row_k] if int(v) == 0 else -d0[row_k]
                return pos if d == 1 else (GRID_W - 1) - pos
        return pos if d0[row_k] == 1 else (GRID_W - 1) - pos

    df["ball_i_norm"] = [_norm_i(p, k) for k, p in enumerate(df["ball_i"])]
    return df


def main(csv_in: str = DEFAULT_CSV_IN,
         csv_out: str = DEFAULT_CSV_OUT,
         meta_out: str = DEFAULT_META_OUT):
    print(f"[grid] in : {csv_in}")
    print(f"[grid] out: {csv_out}")
    df = pd.read_csv(csv_in)

    # 미니맵 크기: game_state.meta.json 의 roi 우선 (영상별 ROI 정확 반영).
    # 기존 버그: 항상 default ROI 크기 사용 → 영상별 미니맵 크기가 다르면
    # 오른쪽 좌표가 i=15 에 뭉개짐.
    mini_w, mini_h = None, None
    gs_meta = Path(csv_in).parent / "game_state.meta.json"
    if gs_meta.exists():
        try:
            roi = json.loads(gs_meta.read_text()).get("roi")
            if roi:
                mini_w, mini_h = roi[2], roi[3]
                print(f"[미니맵] meta roi 사용: {mini_w}x{mini_h}")
        except Exception:
            pass
    if mini_w is None:
        mini_w, mini_h = load_minimap_size()
        print(f"[미니맵] default ROI 사용: {mini_w}x{mini_h} (meta 에 roi 없음)")
    print(f"[그리드] {GRID_W}x{GRID_H}")

    observer = "team" in df.columns and df["team"].notna().any()

    # 프레임별 방향 시계열 (골키퍼 신호 + 하프타임 교대 반영)
    dir_series = estimate_direction_series(df, mini_w=mini_w)
    df["dir_t0"] = dir_series   # xt_model 이 시작팀 시점 추적에 사용

    if observer:
        df = add_grid_columns_observer(df, dir_series, mini_w, mini_h)
        meta = {
            "grid_w": GRID_W, "grid_h": GRID_H,
            "mini_w": mini_w, "mini_h": mini_h,
            "observer": True,
            "direction": "per-frame series (GK 신호)",
        }
    else:
        df = add_grid_columns(df, dir_series, mini_w, mini_h)
        meta = {
            "grid_w": GRID_W, "grid_h": GRID_H,
            "mini_w": mini_w, "mini_h": mini_h,
            "observer": False,
            "direction": "per-frame series (GK 신호)",
            "our_team": 0,
        }

    valid = df[df["ball_i"].notna()]
    print(f"\n[매핑 결과] {len(valid)}/{len(df)} 프레임 그리드 좌표 보유")

    df.to_csv(csv_out, index=False)
    print(f"[저장] {csv_out} (ball_i, ball_j, ball_i_norm, ball_j_norm 컬럼 추가)")

    Path(meta_out).write_text(json.dumps(meta, indent=2))
    print(f"[메타] {meta_out}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    csv_in = args[0] if len(args) > 0 else DEFAULT_CSV_IN
    csv_out = args[1] if len(args) > 1 else (csv_in if csv_in != DEFAULT_CSV_IN else DEFAULT_CSV_OUT)
    meta_out = args[2] if len(args) > 2 else DEFAULT_META_OUT
    from pathlib import Path as _P
    if csv_in != DEFAULT_CSV_IN and len(args) < 3:
        meta_out = str(_P(csv_in).parent / "grid_meta.json")
    main(csv_in, csv_out, meta_out)

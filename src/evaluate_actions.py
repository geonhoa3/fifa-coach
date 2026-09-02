"""
Step 5c — 매 행동의 xT 가치 평가

각 패스/슛에 대해:
  xt_start  = V(출발 셀)
  xt_end    = V(도착 셀)
  xt_change = xt_end - xt_start

양수 = 좋은 행동 (위치 가치 증가)
음수 = 나쁜 행동 (위치 가치 감소)

실행:
  python -m src.evaluate_actions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_CSV_PATH = "data/output/events.csv"
DEFAULT_XT_PATH = "data/output/xt_values.json"


EVAL_OUTCOMES = ["pass_success", "pass_intercepted", "shot", "goal"]

# xt_model.compute_xt 의 goal_zone_i 와 동일해야 함 (골대 영역 시작 i)
GOAL_ZONE_I = 12


def find_previous_ball_idx(df: pd.DataFrame, idx: int) -> int:
    """idx 보다 앞 행 중 ball_i_norm/j_norm 둘 다 있는 가장 가까운 행 인덱스. 없으면 -1."""
    j = idx - 1
    while j >= 0:
        if pd.notna(df.at[j, "ball_i_norm"]) and pd.notna(df.at[j, "ball_j_norm"]):
            return j
        j -= 1
    return -1


def evaluate(csv_path: str = DEFAULT_CSV_PATH, xt_path: str = DEFAULT_XT_PATH):
    print(f"[evaluate] events: {csv_path}")
    print(f"[evaluate] xt    : {xt_path}")
    df = pd.read_csv(csv_path)
    xt_data = json.loads(Path(xt_path).read_text())
    V = np.array(xt_data["V"])
    starts = np.array(xt_data.get("starts", np.ones_like(V)))
    grid_w, grid_h = V.shape

    df["xt_start"] = np.nan
    df["xt_end"] = np.nan
    df["xt_change"] = np.nan

    target_indices = df[df["outcome"].isin(EVAL_OUTCOMES)].index.tolist()
    for idx in target_indices:
        cur_i = df.at[idx, "ball_i_norm"]
        cur_j = df.at[idx, "ball_j_norm"]
        if pd.isna(cur_i) or pd.isna(cur_j):
            continue

        prev_idx = find_previous_ball_idx(df, idx)
        if prev_idx < 0:
            continue

        si = int(df.at[prev_idx, "ball_i_norm"])
        sj = int(df.at[prev_idx, "ball_j_norm"])
        ei = int(cur_i)
        ej = int(cur_j)

        if not (0 <= si < grid_w and 0 <= sj < grid_h):
            continue
        if not (0 <= ei < grid_w and 0 <= ej < grid_h):
            continue

        outcome = df.at[idx, "outcome"]
        is_shot = outcome == "shot"
        is_goal = outcome == "goal"

        # 상대 슛 분리: 행위자 시점 정규화에서 슛 도착이 자기 골대 쪽(i<12)이면
        # 소유권 체크가 샌 것 = 상대가 우리 골문에 쏜 슛 → 평가 제외 + 별도 표시
        # (goal 은 infer_goals 가 소유권 다수결로 이미 our/opponent 분리함 → 면제)
        if is_shot and ei < GOAL_ZONE_I:
            df.at[idx, "outcome"] = "opponent_shot"
            continue

        # 무데이터 셀 제외 — 표본(starts) 0인 셀의 V=0 은 "가치 없음"이 아니라
        # "정보 없음". 그 셀이 출발/도착이면 xt_change 가 극단값으로 튐
        # (슛 비행 조각이 pass 로 쪼개진 아티팩트가 대표 사례) → 평가 제외.
        # 예외 1: 골존(i>=GOAL_ZONE_I) 출발 셀 — xt_model 이 시작점에서 제외하는
        #   영역이라 starts 가 항상 0. 스무딩된 V 값(또는 골존 직전 행)으로 대체.
        #   이걸 스킵하면 골문 앞 침투 패스가 전부 평가 누락됨 (match08 사고).
        # 예외 2: 슛/골의 도착점은 골대 영역(원래 무데이터)이라 도착 셀 검사 면제.
        if starts[si, sj] == 0:
            if si >= GOAL_ZONE_I:
                v_start = float(V[si, sj]) if V[si, sj] > 0 \
                    else float(V[GOAL_ZONE_I - 1, sj])
            elif is_goal:
                v_start = float(V[si, sj])  # 골은 표본 없어도 평가 유지
            else:
                continue
        else:
            v_start = float(V[si, sj])
        if not (is_shot or is_goal) and starts[ei, ej] == 0:
            continue

        v_end = float(V[ei, ej])
        # 슛/골 도착점은 골대 영역(V=0 처리)이므로 강제로 V_max=1.0 부여
        if is_shot or is_goal:
            v_end = 1.0
        df.at[idx, "xt_start"] = v_start
        df.at[idx, "xt_end"] = v_end
        df.at[idx, "xt_change"] = v_end - v_start

    df.to_csv(csv_path, index=False)

    valid = df.dropna(subset=["xt_change"])
    print(f"[평가 완료] 분석된 행동 {len(valid)}개")
    print(f"  평균 xT 변화: {valid['xt_change'].mean():+.3f}")
    print(f"  중앙값      : {valid['xt_change'].median():+.3f}")
    print(f"  양수 (좋음) : {(valid['xt_change'] > 0).sum()}")
    print(f"  음수 (나쁨) : {(valid['xt_change'] < 0).sum()}")
    print(f"  0           : {(valid['xt_change'] == 0).sum()}")

    print(f"\n[outcome 별 평균 xT 변화]")
    for outcome in EVAL_OUTCOMES:
        sub = valid[valid["outcome"] == outcome]
        if len(sub):
            print(f"  {outcome:20s} (n={len(sub):>3d}): "
                  f"평균 {sub['xt_change'].mean():+.3f}")

    print(f"\n[가장 좋은 행동 Top 5]")
    top = valid.nlargest(5, "xt_change")
    for _, r in top.iterrows():
        print(f"  frame {int(r['frame']):>5d} ({r['time_sec']:>5.1f}s) "
              f"{r['outcome']:18s}: V {r['xt_start']:.2f} → {r['xt_end']:.2f} "
              f"({r['xt_change']:+.2f})")

    print(f"\n[가장 나쁜 행동 Top 5]")
    bot = valid.nsmallest(5, "xt_change")
    for _, r in bot.iterrows():
        print(f"  frame {int(r['frame']):>5d} ({r['time_sec']:>5.1f}s) "
              f"{r['outcome']:18s}: V {r['xt_start']:.2f} → {r['xt_end']:.2f} "
              f"({r['xt_change']:+.2f})")

    print(f"\n[저장] {csv_path} (xt_start, xt_end, xt_change 컬럼 추가)")


if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    xt_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_XT_PATH
    evaluate(csv_arg, xt_arg)

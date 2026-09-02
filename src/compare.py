"""
우리 시스템 결과 vs 통계지 정답 비교.

매핑:
  - 슛(우리)        = events.csv 의 outcome == 'shot' 카운트
  - 패스 시도/성공  = pass_success + pass_intercepted (시도), pass_success (성공)
  - 점유율          = our_possession 프레임 / (our_possession + opp_possession) 프레임

실행:
  python -m src.compare
"""

import json
import pandas as pd
from pathlib import Path


GT_PATH = "data/output/ground_truth.json"
EVENTS_PATH = "data/output/events.csv"
OUT_PATH = "data/output/comparison.json"


def load_ground_truth():
    return json.loads(Path(GT_PATH).read_text(encoding="utf-8"))


def compute_our_metrics() -> dict:
    df = pd.read_csv(EVENTS_PATH)
    events = df[df["action"] == "pass_or_shot"]

    shots = int((events["outcome"] == "shot").sum())
    pass_success = int((events["outcome"] == "pass_success").sum())
    pass_intercepted = int((events["outcome"] == "pass_intercepted").sum())
    opponent_pass = int((events["outcome"] == "opponent_pass").sum())
    pass_attempted = pass_success + pass_intercepted
    pass_accuracy = (round(pass_success / pass_attempted * 100)
                     if pass_attempted else 0)

    # 점유율: minimap_valid 프레임 중에서
    #   우리 소유 = ball 검출 + player(빨간 dot) 미검출
    #   상대 소유 = ball 검출 + player 검출
    valid = df[df["minimap_valid"] == True].copy()
    has_ball = valid["ball_x"].notna()
    has_player = valid["player_x"].notna()

    our_frames = int((has_ball & ~has_player).sum())
    opp_frames = int((has_ball & has_player).sum())
    total = our_frames + opp_frames
    possession = round(our_frames / total * 100) if total else 0

    return {
        "shots": shots,
        "passes_attempted": pass_attempted,
        "passes_success": pass_success,
        "passes_intercepted": pass_intercepted,
        "opponent_pass": opponent_pass,
        "pass_accuracy": pass_accuracy,
        "possession": possession,
        "frames_our_poss": our_frames,
        "frames_opp_poss": opp_frames,
    }


def main():
    gt = load_ground_truth()
    ours = compute_our_metrics()

    print("=" * 64)
    print(f"{'지표':<26} | {'정답':>8} | {'우리':>8} | {'오차':>10}")
    print("=" * 64)

    # 슛
    gt_shots = gt["shots"]["us"]
    diff = ours["shots"] - gt_shots
    print(f"{'슛 (우리)':<26} | {gt_shots:>8} | {ours['shots']:>8} | {diff:>+10}")

    # 패스 성공률
    gt_pa = gt["pass_accuracy"]["us"]
    diff = ours["pass_accuracy"] - gt_pa
    print(f"{'패스 성공률 (우리)':<26} | {gt_pa:>7}% | {ours['pass_accuracy']:>7}% | "
          f"{diff:>+9}%p")
    print(f"{'  └ 시도 / 성공':<26} | {' ':>8} | "
          f"{str(ours['passes_attempted']) + '/' + str(ours['passes_success']):>8} |")

    # 점유율
    gt_pos = gt["possession"]["us"]
    diff = ours["possession"] - gt_pos
    print(f"{'점유율 (우리)':<26} | {gt_pos:>7}% | {ours['possession']:>7}% | "
          f"{diff:>+9}%p")
    print(f"{'  └ 우리 / 상대 프레임':<26} | {' ':>8} | "
          f"{str(ours['frames_our_poss']) + '/' + str(ours['frames_opp_poss']):>8} |")

    print("=" * 64)

    # 평가 요약
    print("\n[오차 요약]")
    shot_err_pct = abs(ours["shots"] - gt_shots) / max(gt_shots, 1) * 100
    print(f"  슛 카운트         : 정답 {gt_shots} vs 우리 {ours['shots']} "
          f"→ {shot_err_pct:.0f}% 오차")
    print(f"  패스 성공률       : 정답 {gt_pa}% vs 우리 {ours['pass_accuracy']}% "
          f"→ {abs(ours['pass_accuracy'] - gt_pa)}%p 오차")
    print(f"  점유율            : 정답 {gt_pos}% vs 우리 {ours['possession']}% "
          f"→ {abs(ours['possession'] - gt_pos)}%p 오차")

    # JSON 저장
    out_data = {
        "ground_truth": {"shots": gt_shots, "pass_accuracy": gt_pa, "possession": gt_pos},
        "ours": ours,
        "errors": {
            "shot_count": ours["shots"] - gt_shots,
            "pass_accuracy_pp": ours["pass_accuracy"] - gt_pa,
            "possession_pp": ours["possession"] - gt_pos,
        },
    }
    Path(OUT_PATH).write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[저장] {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
5개 영상에 통합 V_user 적용 → 재평가.

각 영상의 events.csv 에 xt_start/xt_end/xt_change 를
V_user 기준으로 다시 계산해서 덮어씀.

실행:
  python -m src.evaluate_all_with_user
  python -m src.evaluate_all_with_user match04 match05    # 영상 지정
"""

import subprocess
import sys
from pathlib import Path


USER_XT = "data/output/combined/V_user.json"


def discover_videos() -> list:
    """data/output/match*/events.csv 가 있는 영상 자동 탐색."""
    return sorted(p.parent.name for p in Path("data/output").glob("match*/events.csv"))


def main():
    videos = sys.argv[1:] if len(sys.argv) > 1 else discover_videos()

    if not Path(USER_XT).exists():
        print(f"[실패] {USER_XT} 없음. 먼저 src.combine_xt 실행하세요.")
        sys.exit(1)

    for video in videos:
        events = f"data/output/{video}/events.csv"
        if not Path(events).exists():
            print(f"[스킵] {events} 없음")
            continue
        print(f"\n{'=' * 60}")
        print(f"[재평가] {video}")
        print("=" * 60)
        result = subprocess.run([
            "python", "-m", "src.evaluate_actions",
            events, USER_XT,
        ])
        if result.returncode != 0:
            print(f"  [실패] {video}")


if __name__ == "__main__":
    main()

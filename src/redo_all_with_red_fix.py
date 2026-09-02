"""
B 단계 후처리: 빨강 유니폼 fix 적용해서 모든 영상 재처리.

흐름:
  1. 각 영상 폴더의 motion.csv → 새 detect_events 로 events.csv 재생성
  2. evaluate_all_with_user 재실행 (V_user 기반 xt_change 다시)
  3. coaching_report_all 재실행

실행:
  python -m src.redo_all_with_red_fix
"""

import subprocess
import sys
from pathlib import Path


DEFAULT_VIDEOS = ["match04", "match05", "match06", "match07", "match08"]


def main():
    videos = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_VIDEOS

    # 1. detect_events 재실행 (영상별)
    for video in videos:
        motion = f"data/output/{video}/motion.csv"
        events = f"data/output/{video}/events.csv"
        if not Path(motion).exists():
            print(f"[스킵] {motion} 없음")
            continue
        print(f"\n{'=' * 60}")
        print(f"[detect_events 재실행] {video}")
        print("=" * 60)
        subprocess.run(["python", "-m", "src.detect_events", motion, events])

    # 1.5. grid 재실행 (ball_i_norm 컬럼 추가)
    for video in videos:
        events = f"data/output/{video}/events.csv"
        if not Path(events).exists():
            continue
        print(f"\n{'=' * 60}")
        print(f"[grid 재실행] {video}")
        print("=" * 60)
        subprocess.run(["python", "-m", "src.grid", events, events])

    # 2. V_user 기반 재평가
    print(f"\n{'=' * 60}")
    print("[V_user 재평가]")
    print("=" * 60)
    subprocess.run(["python", "-m", "src.evaluate_all_with_user"] + videos)

    # 3. 코칭 리포트 재생성
    print(f"\n{'=' * 60}")
    print("[코칭 리포트 재생성]")
    print("=" * 60)
    subprocess.run(["python", "-m", "src.coaching_report_all"] + videos)


if __name__ == "__main__":
    main()

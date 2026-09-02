"""
모든 영상의 코칭 리포트 일괄 생성.

각 영상 폴더의 events.csv 를 입력으로 받아
같은 폴더에 coaching_report.png 생성.

실행:
  python -m src.coaching_report_all
  python -m src.coaching_report_all match04 match08    # 영상 지정
  python -m src.coaching_report_all --threshold 0.2    # 임계값 조정
"""

import subprocess
import sys
from pathlib import Path


def discover_videos() -> list:
    """data/output/match*/events.csv 가 있는 영상 자동 탐색.
    (pro* 관전 영상은 auto_paths 가 영상 경로를 못 찾아 제외)"""
    return sorted(p.parent.name for p in Path("data/output").glob("match*/events.csv"))


def main():
    args = sys.argv[1:]
    threshold = None
    videos = []
    i = 0
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            threshold = args[i + 1]
            i += 2
        else:
            videos.append(args[i])
            i += 1
    if not videos:
        videos = discover_videos()

    for video in videos:
        events = f"data/output/{video}/events.csv"
        if not Path(events).exists():
            print(f"[스킵] {events} 없음")
            continue
        print(f"\n{'=' * 60}")
        print(f"[코칭 리포트] {video}")
        print("=" * 60)
        cmd = ["python", "-m", "src.coaching_report", events]
        if threshold:
            cmd.append(threshold)
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

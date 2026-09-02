"""
이벤트 시각화 — 몇 개 pass_or_shot 을 미니맵에 화살표 + dot 표시.

격자로 묶어서 한 창에 표시 → 어떤 케이스가 어떻게 분류됐는지 한눈에.

실행:
  python -m src.visualize_events data/videos/match1.mp4
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.minimap import load_roi, crop_minimap, grab_frame


def _parse_dots(s):
    if not isinstance(s, str) or not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def visualize(video_path: str, csv_path: str = "data/output/events.csv",
              n_per_outcome: int = 4):
    df = pd.read_csv(csv_path)
    roi = load_roi()

    # 각 outcome 별로 균등 샘플링
    samples = []
    for outcome in ["pass_success", "pass_intercepted", "shot"]:
        sub = df[(df["action"] == "pass_or_shot") & (df["outcome"] == outcome)]
        if len(sub) > 0:
            samples.append(sub.sample(min(n_per_outcome, len(sub)), random_state=42))
    samples = pd.concat(samples) if samples else df[df["action"] == "pass_or_shot"].head(12)
    samples = samples.sort_values("frame")

    print(f"[시각화] 총 {len(samples)}개 이벤트")

    images = []
    for _, row in samples.iterrows():
        frame_idx = int(row["frame"])
        frame = grab_frame(video_path, frame_idx=frame_idx)
        mini = crop_minimap(frame, roi).copy()
        # 가독성 위해 2배 업스케일
        mini = cv2.resize(mini, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        scale = 2

        # 패스 시작점 = 직전 프레임
        prev_df = df[df["frame"] < frame_idx].dropna(subset=["ball_x", "ball_y"])
        if len(prev_df) > 0:
            prev = prev_df.iloc[-1]
            start = (int(prev["ball_x"] * scale), int(prev["ball_y"] * scale))
            end = (int(row["ball_x"] * scale), int(row["ball_y"] * scale))
            cv2.arrowedLine(mini, start, end, (0, 255, 255), 2, tipLength=0.2)

        # 두 팀 dot
        for dx, dy in _parse_dots(row["t0_dots"]):
            cv2.circle(mini, (int(dx * scale), int(dy * scale)), 4, (255, 255, 0), 1)  # 시안 T0
        for dx, dy in _parse_dots(row["t1_dots"]):
            cv2.circle(mini, (int(dx * scale), int(dy * scale)), 4, (255, 0, 255), 1)  # 마젠타 T1

        # 라벨
        label = f"f{frame_idx} {row['outcome']}"
        cv2.putText(mini, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        images.append(mini)

    if not images:
        print("샘플 없음")
        return

    # 4 x 3 격자 (4열)
    per_row = 4
    h, w = images[0].shape[:2]
    rows = []
    for i in range(0, len(images), per_row):
        chunk = images[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)

    out_path = "data/output/events_grid.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f"[저장] {out_path}")

    win = "Events Grid (any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match1.mp4"
    visualize(video)

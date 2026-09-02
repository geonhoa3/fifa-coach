"""
공 궤적 시각화 — 미니맵 배경 위에 검출된 모든 공 좌표를 점으로 찍는다.
시간 흐름에 따라 색이 파랑 → 빨강 으로 변함 (초기 = 파랑, 후반 = 빨강).
"""
import cv2
import pandas as pd
from pathlib import Path

from src.minimap import load_roi, crop_minimap, grab_frame


def visualize_trajectory(video_path: str,
                          csv_path: str = "data/output/ball_trajectory.csv",
                          bg_frame: int = 9803,
                          save_path: str = "data/output/trajectory.png"):
    df = pd.read_csv(csv_path).dropna(subset=['ball_x', 'ball_y'])
    print(f"[로딩] 검출된 좌표 {len(df)}개")

    bg = crop_minimap(grab_frame(video_path, frame_idx=bg_frame), load_roi()).copy()

    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        ratio = i / max(n - 1, 1)
        # BGR. 시간 따라 파랑(255,0,0) → 빨강(0,0,255)
        color = (int(255 * (1 - ratio)), 0, int(255 * ratio))
        cv2.circle(bg, (int(row['ball_x']), int(row['ball_y'])), 1, color, -1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, bg)
    print(f"[저장] {save_path}")

    win = "Ball Trajectory (any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match1.mp4"
    visualize_trajectory(video)

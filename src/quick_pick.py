"""
1클릭 미니맵 ROI 잡기.

사용 흐름:
  1. 영상 한 프레임 띄움
  2. 사용자가 미니맵 가운데 한 번 클릭
  3. 클릭 위치 주변 400x300 영역에서 dot 자동 검출
  4. dot 들의 bounding box + 표준 비율(1.7:1) 강제
  5. 시각화로 박스 보여줌 → ENTER 저장, r 다시 클릭

실행:
    python -m src.quick_pick data/videos/foo.mp4 [frame_idx]
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from src.minimap import save_roi, grab_frame
from src.detect_teams import GRASS_LOW, GRASS_HIGH, get_dot_pixels


def find_dots_around(frame, cx: int, cy: int,
                      box_w_frac: float = 0.115,
                      aspect: float = 1.7):
    """클릭 위치 ± 검색 영역에서 dot centroid 자동 보정 + 표준 박스.

    1. 클릭 주변 (박스의 1.5배) 검색 영역에서 dot 검출
    2. dot 8개 이상이면 centroid 로 박스 중심 보정
    3. 표준 비율 박스로 그림 (영상 폭의 11.5%)
    → 사용자가 미니맵 정중앙 못 맞춰도 자동 보정됨
    """
    h, w = frame.shape[:2]
    # 표준 박스 크기 (4K 영상 약 295×173)
    box_w = int(w * box_w_frac)
    box_h = int(box_w / aspect)

    # 검색 영역 = 박스의 1.5배 (클릭 부정확 흡수)
    sw = int(box_w * 1.5)
    sh = int(box_h * 1.5)
    sx_min = max(0, cx - sw // 2)
    sy_min = max(0, cy - sh // 2)
    sx_max = min(w, cx + sw // 2)
    sy_max = min(h, cy + sh // 2)

    sub = frame[sy_min:sy_max, sx_min:sx_max]
    # detect_teams 의 get_dot_pixels 사용 (흰 ring dot 보호 + 잔디/노란/빨강/검정 제외)
    _, keep = get_dot_pixels(sub)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (1 <= a <= 80):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        dx = int(M["m10"] / M["m00"])
        dy = int(M["m01"] / M["m00"])
        dots.append((dx + sx_min, dy + sy_min))

    # dot 8개 이상이면 보정. bounding box 중심 사용 (centroid 보다 정확).
    # dot 분포 한쪽 치우침 무관하게 미니맵 가운데 잡힘.
    if len(dots) >= 8:
        xs = [d[0] for d in dots]
        ys = [d[1] for d in dots]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2

    # 보정된 중심 기준 표준 박스
    x_min = max(0, cx - box_w // 2)
    y_min = max(0, cy - box_h // 2)
    x_max = min(w, x_min + box_w)
    y_max = min(h, y_min + box_h)
    return (x_min, y_min, x_max - x_min, y_max - y_min), len(dots)


def quick_pick_roi(video_path: str, frame_idx: int = None):
    """1클릭 ROI 잡기 UI."""
    if frame_idx is None:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = n_frames // 3

    frame = grab_frame(video_path, frame_idx, verbose=False)

    state = {"click": None, "roi": None, "n_dots": 0}

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        state["click"] = (x, y)
        roi, n_dots = find_dots_around(frame, x, y)
        state["roi"] = roi
        state["n_dots"] = n_dots
        if roi is None:
            print(f"[클릭] ({x}, {y}) → dot {n_dots}개 (부족, 다시 시도)")
        else:
            print(f"[클릭] ({x}, {y}) → dot {n_dots}개 → ROI {roi}")

    win = "Click minimap center | ENTER=save, r=reset, ESC=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(win, on_click)

    print("[안내] 영상에서 미니맵 가운데를 한 번 클릭하세요")
    print("       잘못 클릭하면 r 키로 리셋, 잘 잡혔으면 ENTER 로 저장")

    while True:
        viz = frame.copy()
        if state["roi"]:
            x, y, rw, rh = state["roi"]
            cv2.rectangle(viz, (x, y), (x + rw, y + rh), (0, 255, 0), 4)
            cv2.putText(viz, f"ROI {rw}x{rh} ({state['n_dots']} dots)",
                        (x, max(30, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if state["click"]:
            cx, cy = state["click"]
            cv2.drawMarker(viz, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 30, 3)
        cv2.putText(viz, "ENTER=save  r=reset  ESC=cancel",
                    (20, viz.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow(win, viz)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k == ord('q'):
            cv2.destroyAllWindows()
            print("[취소]")
            return None
        if k == ord('r'):
            state["click"] = None
            state["roi"] = None
            print("[리셋]")
        if k == 13:
            if state["roi"] is None:
                print("[대기] 먼저 미니맵 가운데 클릭")
                continue
            cv2.destroyAllWindows()
            save_roi(state["roi"], video_path=video_path)
            print(f"[저장] ROI = {state['roi']}")
            return state["roi"]


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match1.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    quick_pick_roi(video, frame_idx=frame_idx)

"""
Step 3a — 빨강 dot (조작 선수 = 공 소유 우리팀 선수) 검출

핵심: HSV에서 빨강은 H 값이 0과 180 양 끝에 걸쳐있어서
       두 구간(0~10, 170~180)을 따로 마스킹 후 합쳐야 한다.

실행:
  python -m src.detect_player data/videos/match1.mp4 9803    # 한 프레임 시각화
"""

import cv2
import numpy as np

from src.minimap import load_roi, crop_minimap, grab_frame
from src.detect_ball import detect_ball


# HSV 빨강 범위 (두 구간). S/V 임계값은 진한 빨강만 잡도록.
RED_LOW1 = np.array([0, 120, 100])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 100])
RED_HIGH2 = np.array([180, 255, 255])

MIN_AREA = 5


def red_mask(mini_bgr):
    """미니맵에서 빨강 픽셀만 추출한 마스크."""
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    m2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    return cv2.bitwise_or(m1, m2)


def detect_red_dot(mini_bgr):
    """가장 큰 빨강 영역의 무게중심 = 조작 선수 좌표. 검출 실패 시 None."""
    mask = red_mask(mini_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return None
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def debug_single_frame(video_path: str, frame_idx: int = 9803):
    roi = load_roi()
    frame = grab_frame(video_path, frame_idx=frame_idx)
    mini = crop_minimap(frame, roi)

    ball = detect_ball(mini)
    player = detect_red_dot(mini)
    print(f"[검출] 공: {ball}, 조작 선수: {player}")

    mask = red_mask(mini)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    viz = mini.copy()
    if ball:
        cv2.circle(viz, ball, 6, (0, 255, 255), 2)   # 노랑: 공
    if player:
        cv2.circle(viz, player, 8, (0, 0, 255), 2)   # 빨강: 조작 선수

    combo = np.hstack([mini, mask_bgr, viz])
    win = "Original | Red Mask | Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, combo)
    print("[안내] 아무 키나 누르면 닫힘")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match1.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 9803
    debug_single_frame(video, frame_idx)

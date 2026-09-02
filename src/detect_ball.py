"""
Step 2 — 노란 십자(공) 검출

매 프레임 미니맵에서 노란 십자가의 무게중심을 찾아 공 좌표로 사용.
HSV 마스킹 → contour → centroid 의 표준 패턴.

실행:
  python -m src.detect_ball debug data/videos/match1.mp4 9803      # 한 프레임 시각화
  python -m src.detect_ball process data/videos/match1.mp4         # 전체 영상 → CSV
  python -m src.detect_ball process data/videos/match1.mp4 3       # 3프레임마다 (속도 ↑)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.minimap import load_roi, crop_minimap, grab_frame


# HSV 노란색 범위. 영상마다 다르면 debug 모드 결과 보고 튜닝.
YELLOW_LOW = np.array([20, 100, 150])
YELLOW_HIGH = np.array([35, 255, 255])

# 노란 영역이 이보다 작으면 노이즈로 무시 (픽셀 수)
MIN_AREA = 5


def detect_ball(mini_bgr):
    """
    잘라낸 미니맵 이미지에서 노란 십자(공) 위치 반환.
    검출 실패하면 None.
    """
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return None

    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def debug_single_frame(video_path: str, frame_idx: int = 9803):
    """
    한 프레임만 처리해서 결과 시각화.
    HSV 임계값이 잘 맞는지 확인할 때 사용.
    """
    roi = load_roi()
    frame = grab_frame(video_path, frame_idx=frame_idx)
    mini = crop_minimap(frame, roi)

    ball = detect_ball(mini)
    print(f"[검출] 공 좌표: {ball}")

    # 마스크 시각화 (원본 | 마스크 | 검출 표시)
    hsv = cv2.cvtColor(mini, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    viz = mini.copy()
    if ball:
        cv2.circle(viz, ball, 10, (0, 0, 255), 2)
        cv2.putText(viz, f"{ball}", (ball[0] + 12, ball[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    combo = np.hstack([mini, mask_bgr, viz])

    win = "Original | Mask | Detection (any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, combo)
    print("[안내] 창이 보이면 아무 키나 누르면 닫힘")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path: str,
                  out_csv: str = "data/output/ball_trajectory.csv",
                  sample_every: int = 1):
    """
    영상 전체 → 프레임별 공 좌표 시계열 CSV.

    sample_every=N 이면 N프레임마다 1번씩 (속도 ↑, 정밀도 ↓).
    """
    roi = load_roi()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[처리] 총 {n_frames}프레임, {fps:.1f}fps, sample_every={sample_every}")

    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            mini = crop_minimap(frame, roi)
            ball = detect_ball(mini)
            records.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "ball_x": ball[0] if ball else None,
                "ball_y": ball[1] if ball else None,
                "detected": ball is not None,
            })

        frame_idx += 1
        if frame_idx % 1000 == 0:
            pct = frame_idx * 100 // n_frames
            print(f"  진행 {frame_idx}/{n_frames} ({pct}%)")

    cap.release()

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    detected = int(df["detected"].sum())
    total = len(df)
    print(f"\n[완료] {total}프레임 중 {detected}프레임 검출 ({detected*100//total}%)")
    print(f"[저장] {out_csv}")


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "debug"
    video = sys.argv[2] if len(sys.argv) > 2 else "data/videos/match1.mp4"

    if cmd == "debug":
        frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 9803
        debug_single_frame(video, frame_idx=frame_idx)

    elif cmd == "process":
        sample_every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        process_video(video, sample_every=sample_every)

    else:
        print("usage: python -m src.detect_ball [debug|process] <video> [frame_idx|sample_every]")

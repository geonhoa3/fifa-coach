"""
미니맵 자동 검출 — 영상에서 미니맵 ROI 를 자동으로 찾고 사용자가 검증.

알고리즘:
1. 영상 10시점 샘플링
2. 각 프레임에서 잔디 영역 connected components 추출
3. 미니맵 후보 점수 (형태 비율 1.2~2.2, 화면의 3~20% 크기, 내부 dot 패턴)
4. 여러 프레임에서 일관된 위치 → ROI 확정
5. 사용자에게 시각화 보여주고 ENTER (수락) / ESC (manual 모드)

실행:
    python -m src.auto_minimap data/videos/foo.mp4
"""

import sys
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

from src.minimap import save_roi, grab_frame
from src.detect_teams import GRASS_LOW, GRASS_HIGH


def _count_small_dots(bbox_bgr):
    """ROI 내 작은 dot 개수 (실제 미니맵 dot 카운트)."""
    hsv = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    non_grass = cv2.bitwise_not(grass)
    contours, _ = cv2.findContours(non_grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_dots = 0
    for c in contours:
        a = cv2.contourArea(c)
        if 1 <= a <= 80:
            small_dots += 1
    return small_dots


def find_minimap_candidates(frame, min_area: int = 5000, max_area_pct: float = 0.10):
    """한 프레임에서 미니맵 후보 ROI 들 반환 (x, y, w, h, score).

    개선 포인트:
    - max_area_pct 0.10 (이전 0.20) — 메인 필드 (큰 잔디 영역) 자동 배제
    - min_area 5000 — 너무 작은 잡음 영역 배제
    - 형태 비율 1.4~1.8 (실제 미니맵 비율 1.6 근처)
    - 작은 dot 10개 이상 검증 (메인 필드 잡힘 방지)
    - 위치 가중치 (보통 하단 1/3 또는 코너)
    """
    h, w = frame.shape[:2]
    max_area = int(w * h * max_area_pct)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)

    # close 로 dot 틈 메우기 (connected components 가 큰 덩어리로)
    kernel = np.ones((9, 9), np.uint8)
    grass_filled = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, kernel)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(grass_filled, 8)

    candidates = []
    for i in range(1, nlabels):
        x, y, cw, ch, area = stats[i]
        if area < min_area or area > max_area:
            continue
        aspect = cw / max(ch, 1)
        if not (1.4 <= aspect <= 1.8):  # 미니맵 비율 좁힘
            continue

        bbox_grass = grass[y:y + ch, x:x + cw]
        grass_ratio = (bbox_grass > 0).sum() / max(cw * ch, 1)
        if not (0.4 <= grass_ratio <= 0.85):  # 너무 잔디만이면 메인 필드, 너무 적으면 미니맵 아님
            continue

        # 작은 dot 명시 카운트 (메인 필드는 dot 거의 없음)
        bbox = frame[y:y + ch, x:x + cw]
        n_dots = _count_small_dots(bbox)
        if n_dots < 10:
            continue

        # 위치 가중치 (보통 하단 또는 코너에 미니맵 있음)
        cy = y + ch / 2
        loc_score = max(0.0, (cy / h - 0.5) * 2)  # 0(상단)~1(하단)

        # 점수 종합
        shape_score = 1.0 - abs(aspect - 1.6) / 0.4
        dot_score = min(n_dots / 22.0, 1.0)  # 22개 = 양팀 11명 만점
        score = dot_score * 0.45 + grass_ratio * 0.2 + shape_score * 0.2 + loc_score * 0.15

        candidates.append((int(x), int(y), int(cw), int(ch), float(score)))

    return candidates


def find_minimap_by_dots(frame, grid: int = 40, pad: int = 30,
                          y_min_frac: float = 0.60,
                          max_w_frac: float = 0.30,
                          max_h_frac: float = 0.20):
    """dot 밀집도 기반 미니맵 검출.

    추가 제한:
    - y_min_frac: dot 이 화면 하단 60% 이상에 있어야 (메인 필드 가운데 배제)
    - max_w/h_frac: ROI 크기 상한 (메인 필드 잡힘 방지)
    - 그리드 30px + 윈도우 3x2 (실제 미니맵 비율에 맞춤)
    """
    h, w = frame.shape[:2]
    y_min_px = int(h * y_min_frac)
    max_w_px = int(w * max_w_frac)
    max_h_px = int(h * max_h_frac)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    grass_dilated = cv2.dilate(grass, np.ones((5, 5), np.uint8), iterations=2)

    non_grass = cv2.bitwise_not(grass)
    contours, _ = cv2.findContours(non_grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_centers = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (1 <= a <= 40):
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # 화면 하단에만 (미니맵 일반 위치)
        if cy < y_min_px:
            continue
        # 잔디 영역 안
        if grass_dilated[cy, cx] == 0:
            continue
        dot_centers.append((cx, cy))

    if len(dot_centers) < 10:
        return None, 0

    # 30px 그리드 양자화 + 3x2 윈도우 (미니맵 비율)
    grid_count = {}
    for cx, cy in dot_centers:
        key = (cx // grid, cy // grid)
        grid_count[key] = grid_count.get(key, 0) + 1

    best = None
    best_count = 0
    for (gx, gy) in grid_count:
        wcount = sum(grid_count.get((gx + dx, gy + dy), 0)
                     for dx in range(-3, 4) for dy in range(-2, 3))
        if wcount > best_count:
            best_count = wcount
            best = (gx, gy)

    if best is None or best_count < 10:
        return None, 0

    # best 그리드 중심 픽셀 좌표
    gx, gy = best
    gx_px = (gx + 0.5) * grid
    gy_px = (gy + 0.5) * grid
    # 픽셀 거리 R 안의 dot 모두 포함. R 크게 잡고 max_w_frac 으로 안전 확보.
    R = 250
    # 가로 더 넓게 (미니맵 비율 1.6:1) — 타원형 거리 사용
    Rx, Ry = R, int(R * 0.7)
    nearby = [(cx, cy) for cx, cy in dot_centers
              if ((cx - gx_px) / Rx) ** 2 + ((cy - gy_px) / Ry) ** 2 <= 1.0]
    if len(nearby) < 10:
        return None, 0

    xs = [c[0] for c in nearby]
    ys = [c[1] for c in nearby]
    x_min = max(0, min(xs) - pad)
    y_min = max(0, min(ys) - pad)
    x_max = min(w, max(xs) + pad)
    y_max = min(h, max(ys) + pad)
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    # 비율 1.7:1 강제 확장 (검출 안 된 미니맵 끝 dot 까지 박스 확장)
    target_aspect = 1.7
    if roi_w / max(roi_h, 1) < target_aspect:
        target_w = int(roi_h * target_aspect)
        cx_mid = (x_min + x_max) // 2
        x_min = max(0, cx_mid - target_w // 2)
        x_max = min(w, cx_mid + target_w // 2)
        roi_w = x_max - x_min
    elif roi_h * 1.0 / max(roi_w, 1) < 1 / 2.2:
        target_h = int(roi_w / 2.0)
        cy_mid = (y_min + y_max) // 2
        y_min = max(0, cy_mid - target_h // 2)
        y_max = min(h, cy_mid + target_h // 2)
        roi_h = y_max - y_min

    # 크기 제한
    if roi_w > max_w_px or roi_h > max_h_px:
        return None, 0
    if not (1.2 <= roi_w / max(roi_h, 1) <= 2.2):
        return None, 0

    return (x_min, y_min, roi_w, roi_h), best_count


def auto_detect_roi(video_path: str, n_samples: int = 10):
    """여러 프레임에서 일관된 미니맵 위치 찾기 (dot 밀집도 기반)."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[자동 검출] {n_samples}시점 샘플링 ({n_frames}프레임 영상)")

    all_candidates = []
    for i in range(n_samples):
        idx = int(n_frames * (i + 1) / (n_samples + 1))
        try:
            frame = grab_frame(video_path, idx, verbose=False)
        except Exception as e:
            print(f"  [경고] {idx}번째 프레임 읽기 실패: {e}")
            continue
        roi_and_count = find_minimap_by_dots(frame)
        roi, count = roi_and_count
        if roi is None:
            continue
        x, y, cw, ch = roi
        # 20px bin 단위로 quantize (위치 안정성 검사)
        bx, by, bw, bh = x // 20, y // 20, cw // 20, ch // 20
        all_candidates.append(((bx, by, bw, bh), float(count), (x, y, cw, ch)))

    if not all_candidates:
        print("[실패] 미니맵 후보 없음")
        return None

    # 빈도 + 점수 합산
    agg = {}
    for key, score, raw in all_candidates:
        if key not in agg:
            agg[key] = [0, 0.0, []]
        agg[key][0] += 1
        agg[key][1] += score
        agg[key][2].append(raw)

    # 가장 많이 등장한 위치 + 평균 점수 (최소 2시점 이상)
    best_key = None
    best_metric = -1
    for key, (count, total_score, raws) in agg.items():
        if count < 2:
            continue
        metric = count * (total_score / count)
        if metric > best_metric:
            best_metric = metric
            best_key = key

    if best_key is None:
        # 2시점 미만이라도 가장 점수 높은 것
        best_key = max(agg.keys(), key=lambda k: agg[k][1])

    count, total_score, raws = agg[best_key]
    avg = np.array(raws).mean(axis=0).astype(int)
    print(f"[발견] ROI=(x={avg[0]}, y={avg[1]}, w={avg[2]}, h={avg[3]}), "
          f"{count}/{n_samples}시점 일치")
    return tuple(int(v) for v in avg)


def visualize_roi(video_path: str, roi: tuple, frame_idx: int = None):
    """검출된 ROI 를 영상 위에 그려서 반환."""
    if frame_idx is None:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = n_frames // 3

    frame = grab_frame(video_path, frame_idx, verbose=False)
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.putText(frame, f"Auto ROI ({w}x{h})", (x, max(30, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, "ENTER=accept / ESC=manual",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame


def auto_setup(video_path: str) -> bool:
    """미니맵 자동 검출 + 사용자 검증. True=수락, False=거부."""
    roi = auto_detect_roi(video_path)
    if roi is None:
        print("[안내] manual calibrate 사용:")
        print(f"       python -m src.minimap calibrate {video_path}")
        return False

    viz = visualize_roi(video_path, roi)
    win = "Auto-detected minimap"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, viz)
    print("[검증] 녹색 박스가 미니맵 위에 잘 잡혔으면 ENTER, 아니면 ESC")

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 13:
            cv2.destroyAllWindows()
            save_roi(roi, video_path=video_path)
            print("[수락] ROI 저장 완료")
            return True
        if k == 27:
            cv2.destroyAllWindows()
            print("[거부] manual calibrate 사용:")
            print(f"       python -m src.minimap calibrate {video_path}")
            return False


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match1.mp4"
    auto_setup(video)

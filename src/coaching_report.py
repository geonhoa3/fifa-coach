"""
Step 5d — 자동 코칭 리포트

흐름:
  1. events.csv 에서 |xt_change| > THRESHOLD 인 행동만 필터링
  2. 각 행동을 카테고리 + 자연어 코멘트로 라벨링
  3. 미니맵 + 화살표 + 정보 카드 렌더링
  4. 격자로 묶어 한 장 PNG + 콘솔 텍스트 리포트

실행:
  python -m src.coaching_report
  python -m src.coaching_report 0.2          # 임계값 조정
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from src.minimap import load_roi, crop_minimap, grab_frame
from src.detect_teams import is_minimap_valid, get_dot_pixels


def is_strict_valid(mini_bgr, grass_threshold: float = 0.5,
                     min_dot_count: int = 10) -> bool:
    """
    잔디 비율 OK + 작은 dot 개수 충족.
    셀럽 화면은 큰 색 영역 1~2개 → dot 개수 부족.
    미니맵은 작은 dot 16~22개.
    """
    if not is_minimap_valid(mini_bgr, grass_threshold):
        return False
    _, keep = get_dot_pixels(mini_bgr)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    # 작은 dot 크기 (3~50 픽셀) 인 영역만 카운트
    valid_dots = [c for c in contours if 3 <= cv2.contourArea(c) <= 50]
    return len(valid_dots) >= min_dot_count


DEFAULT_CSV_IN = "data/output/events.csv"
DEFAULT_VIDEO = "data/videos/match02.mp4"
DEFAULT_OUT_PNG = "data/output/coaching_report.png"
DEFAULT_THRESHOLD = 0.3


def auto_paths(csv_path: str) -> tuple[str, str]:
    """events.csv 경로에서 영상명 추론 → 영상 경로 + 출력 PNG 경로."""
    csv_p = Path(csv_path)
    parent_name = csv_p.parent.name  # 예: match08
    if parent_name in ("output", ".") or not parent_name.startswith("match"):
        return DEFAULT_VIDEO, DEFAULT_OUT_PNG
    video = f"data/videos/{parent_name}.mp4"
    out = str(csv_p.parent / "coaching_report.png")
    return video, out


CATEGORIES = {
    "GOAL":        ("골! 공격 마무리 성공", (0, 255, 0)),
    "GOOD_SHOT":   ("좋은 슛 시도", (0, 255, 0)),
    "GOOD_PASS":   ("좋은 전진 패스", (0, 200, 0)),
    "BAD_RETREAT": ("좋은 위치에서 후방 패스 (기회 놓침)", (0, 165, 255)),
    "RISKY_PASS":  ("도전적 패스 (인터셉트당함)", (0, 255, 255)),
    "BAD_PASS":    ("나쁜 위치로 패스 (인터셉트당함)", (0, 0, 255)),
}


def categorize(outcome: str, xt_change: float) -> str:
    if outcome == "goal":
        return "GOAL"
    if outcome == "shot":
        return "GOOD_SHOT" if xt_change > 0.5 else "GOOD_SHOT"
    if outcome == "pass_success":
        if xt_change > 0.3:
            return "GOOD_PASS"
        if xt_change < -0.3:
            return "BAD_RETREAT"
        return "GOOD_PASS"  # 약한 양수
    if outcome == "pass_intercepted":
        if xt_change > 0.3:
            return "RISKY_PASS"
        return "BAD_PASS"
    return "GOOD_PASS"


def render_card(video_path: str, df: pd.DataFrame, idx: int,
                 prev_idx: int, roi: tuple, scale: int = 2):
    row = df.loc[idx]
    prev = df.loc[prev_idx]

    cat = categorize(row["outcome"], row["xt_change"])
    comment, color = CATEGORIES[cat]

    frame = grab_frame(video_path, frame_idx=int(row["frame"]))
    mini_raw = crop_minimap(frame, roi)
    # 미니맵 보이는 프레임이면 렌더 (dot 검출은 YOLO가 따로 함 — 잔디만 체크)
    if mini_raw.size == 0 or not is_minimap_valid(mini_raw):
        return None
    mini = mini_raw.copy()
    mini = cv2.resize(mini, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)

    start = (int(prev["ball_x"] * scale), int(prev["ball_y"] * scale))
    end = (int(row["ball_x"] * scale), int(row["ball_y"] * scale))
    cv2.arrowedLine(mini, start, end, color, 3, tipLength=0.25)

    h, w = mini.shape[:2]
    info = np.full((110, w, 3), 25, dtype=np.uint8)

    time_str = f"{int(row['time_sec'] // 60)}:{int(row['time_sec'] % 60):02d}"
    cv2.putText(info, f"{time_str}  [{cat}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(info,
                f"xT {row['xt_start']:.2f} -> {row['xt_end']:.2f} ({row['xt_change']:+.2f})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(info, row["outcome"],
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return np.vstack([mini, info])


def main(threshold: float = DEFAULT_THRESHOLD,
         csv_in: str = DEFAULT_CSV_IN,
         video: str = None,
         out_png: str = None):
    if video is None or out_png is None:
        auto_video, auto_out = auto_paths(csv_in)
        video = video or auto_video
        out_png = out_png or auto_out

    print(f"[코칭 리포트]")
    print(f"  events: {csv_in}")
    print(f"  video : {video}")
    print(f"  output: {out_png}")

    df = pd.read_csv(csv_in)
    sig = df.dropna(subset=["xt_change"]).copy()
    # 골/슛은 그 자체가 최상위 이벤트 — xt_change 임계값과 무관하게 항상 포함
    # (골존 근처 출발 골은 xt_change 가 +0.2대라 임계값 0.3에 걸러졌던 문제)
    sig = sig[(sig["xt_change"].abs() > threshold)
              | (sig["outcome"].isin(["goal", "shot"]))].sort_values("frame")
    print(f"  의미 있는 행동 {len(sig)}개 (|xt_change| > {threshold} 또는 골/슛)")

    roi = load_roi(video_path=video)   # 영상별 ROI (default 오크롭 방지)
    cards = []

    print("\n[행동 목록]")
    for idx in sig.index:
        prev_idx = idx - 1
        while prev_idx >= 0 and pd.isna(df.at[prev_idx, "ball_x"]):
            prev_idx -= 1
        if prev_idx < 0:
            continue

        row = df.loc[idx]
        cat = categorize(row["outcome"], row["xt_change"])
        comment, _ = CATEGORIES[cat]
        time_str = f"{int(row['time_sec'] // 60)}:{int(row['time_sec'] % 60):02d}"
        print(f"  {time_str} [{cat:11s}] xT {row['xt_start']:.2f}->{row['xt_end']:.2f} "
              f"({row['xt_change']:+.2f}) - {comment}")

        try:
            card = render_card(video, df, idx, prev_idx, roi)
            if card is None:
                print(f"    (미니맵 valid 실패 - 스킵)")
                continue
            cards.append(card)
        except Exception as e:
            print(f"    렌더 실패: {e}")

    if not cards:
        print("렌더된 카드 없음")
        return

    # 격자 (최대 4열) — 빈 칸이 최소가 되도록 열 수 조정
    # (카드 1개=1열, 5개=3+2 배치 등. 검은 여백 카드가 혼란을 준다는 피드백)
    import math
    n_rows = math.ceil(len(cards) / 4)
    per_row = math.ceil(len(cards) / n_rows)
    h, w = cards[0].shape[:2]
    rows = []
    for i in range(0, len(cards), per_row):
        chunk = cards[i:i + per_row]
        while len(chunk) < per_row:
            chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, grid)
    print(f"\n[저장] {out_png}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    csv_in = DEFAULT_CSV_IN
    thr = DEFAULT_THRESHOLD
    video = None
    out_png = None
    for a in args:
        if a.endswith(".csv"):
            csv_in = a
        elif a.endswith(".mp4"):
            video = a            # 분석 대상 영상 직접 지정 (배경 미니맵 출처)
        elif a.endswith(".png"):
            out_png = a
        else:
            try:
                thr = float(a)
            except ValueError:
                pass
    main(threshold=thr, csv_in=csv_in, video=video, out_png=out_png)

"""
V_user vs V_pro 비교 — 사용자 영상의 V와 고수 영상의 V를 비교.

출력:
  - 3-panel 히트맵: User, Pro, Diff
  - "사용자가 가장 부족한 셀" 통계 (어디서 코칭 필요한지)

실행:
  python -m src.compare_xt
"""

import json
import sys
import numpy as np
import cv2
from pathlib import Path


USER_PATH = "data/output/combined/V_user.json"
PRO_PATH = "data/output/combined/V_pro.json"
OUT_PATH = "data/output/combined/V_compare.png"


def load_v(path: str):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    return np.array(data["V"]), np.array(data["starts"])


def render_value_panel(V, scale: int = 50):
    """V 행렬 → 히트맵 (파랑→빨강 그라데이션)."""
    grid_w, grid_h = V.shape
    img = np.zeros((grid_h * scale, grid_w * scale, 3), dtype=np.uint8)
    vmax = V.max() if V.max() > 0 else 1
    for i in range(grid_w):
        for j in range(grid_h):
            v = max(min(V[i, j] / vmax, 1), 0)
            r = int(v * 255)
            b = 255 - r
            cv2.rectangle(img, (i * scale, j * scale),
                          ((i + 1) * scale, (j + 1) * scale), (b, 30, r), -1)
            cv2.putText(img, f"{V[i, j]:.2f}",
                        (i * scale + 4, j * scale + scale - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def render_diff_panel(diff, scale: int = 50):
    """diff → 히트맵. 음수=빨강(사용자 부족), 양수=녹색(사용자 우수)."""
    grid_w, grid_h = diff.shape
    img = np.zeros((grid_h * scale, grid_w * scale, 3), dtype=np.uint8)
    for i in range(grid_w):
        for j in range(grid_h):
            d = diff[i, j]
            if d < -0.05:
                r = min(int(abs(d) * 510), 255)
                color = (50, 50, r)
            elif d > 0.05:
                g = min(int(d * 510), 255)
                color = (50, g, 50)
            else:
                color = (60, 60, 60)
            cv2.rectangle(img, (i * scale, j * scale),
                          ((i + 1) * scale, (j + 1) * scale), color, -1)
            cv2.putText(img, f"{d:+.2f}",
                        (i * scale + 2, j * scale + scale - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def add_label(img, text: str, height: int = 40):
    label = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([label, img])


def main():
    user = load_v(USER_PATH)
    pro = load_v(PRO_PATH)

    if user is None:
        print(f"[실패] {USER_PATH} 없음. 먼저 V_user 생성하세요.")
        sys.exit(1)
    if pro is None:
        print(f"[실패] {PRO_PATH} 없음.")
        print(f"  고수 영상 처리 후:")
        print(f"  python -m src.combine_xt pro01 pro02 ... --output V_pro")
        sys.exit(1)

    V_user, starts_user = user
    V_pro, starts_pro = pro

    if V_user.shape != V_pro.shape:
        print(f"[실패] shape 불일치: user {V_user.shape} vs pro {V_pro.shape}")
        sys.exit(1)

    grid_w, grid_h = V_user.shape
    diff = V_user - V_pro

    # 통계
    print(f"\n[비교 통계]")
    user_avg = V_user[V_user > 0].mean() if (V_user > 0).any() else 0
    pro_avg = V_pro[V_pro > 0].mean() if (V_pro > 0).any() else 0
    print(f"  V_user 평균 (0 제외): {user_avg:.3f}")
    print(f"  V_pro 평균 (0 제외) : {pro_avg:.3f}")
    print(f"  Diff 평균           : {diff.mean():+.3f}")
    print(f"  사용자 부족 셀 (-0.1↓): {int((diff < -0.1).sum())}/{grid_w * grid_h}")
    print(f"  사용자 우수 셀 (+0.1↑): {int((diff > 0.1).sum())}/{grid_w * grid_h}")

    # 가장 부족한 셀
    print(f"\n[가장 부족한 셀 Top 5] (사용자가 프로 대비 떨어지는 위치)")
    flat = []
    for i in range(grid_w):
        for j in range(grid_h):
            if starts_user[i, j] >= 5 and starts_pro[i, j] >= 5:  # 둘 다 데이터 충분
                flat.append((diff[i, j], i, j))
    flat.sort()
    for d, i, j in flat[:5]:
        print(f"  i={i:2d}, j={j:2d}: 사용자 {V_user[i, j]:.2f} vs 프로 {V_pro[i, j]:.2f} "
              f"(차이 {d:+.2f})")

    # 시각화: 세로로 3개 panel
    panel_user = add_label(render_value_panel(V_user), "V_user (you)")
    panel_pro = add_label(render_value_panel(V_pro), "V_pro (pros)")
    panel_diff = add_label(render_diff_panel(diff),
                           "Diff (red = behind pros, green = ahead)")
    combo = np.vstack([panel_user, panel_pro, panel_diff])

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_PATH, combo)
    print(f"\n[저장] {OUT_PATH}")


if __name__ == "__main__":
    main()

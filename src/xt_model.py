"""
Step 5b — xT 가치맵 (A 방식: 골대 영역 도달 기반)

V(i, j) = (i, j) 셀에서 시작한 점유가 N초 안에
          상대 골대 영역(i_norm >= goal_zone_i) 에 도달한 비율.

값이 높을수록 위험한 위치(=공격에 좋음).

실행:
  python -m src.xt_model
"""

import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


CSV_IN = "data/output/events.csv"
META_IN = "data/output/grid_meta.json"
OUT_JSON = "data/output/xt_values.json"
OUT_HEATMAP = "data/output/xt_heatmap.png"


def compute_xt(window_seconds: int = 10, sample_every: int = 3,
                fps: int = 30, goal_zone_i: int = 12,
                fold: bool = True, smooth_sigma: float = 0.8):
    """
    각 셀의 V(i,j) 계산.

    fold=True: 좌우(j축) 대칭 folding — 필드는 상하 대칭이므로
        j와 (grid_h-1-j) 셀의 카운트를 합산 → 셀당 표본 2배.
    smooth_sigma>0: starts/reached 카운트에 가우시안 스무딩 후 나눔 —
        인접 셀과 표본 공유 → 희소 셀(골대 근처 등) 노이즈 완화.
        비율(V)이 아니라 카운트를 스무딩해야 표본 수가 가중치로 반영됨.

    Returns:
        V: (grid_w, grid_h) float 행렬
        starts: 셀별 시작 횟수 (fold 반영, 스무딩 전 정수)
        reached: 셀별 골대 영역 도달 횟수 (동일)
    """
    df = pd.read_csv(CSV_IN)
    meta = json.loads(Path(META_IN).read_text())
    grid_w = meta["grid_w"]
    grid_h = meta["grid_h"]

    # 윈도우 = N초 안의 행 수 (sample_every 고려)
    window_rows = int(window_seconds * fps / sample_every)

    starts = np.zeros((grid_w, grid_h), dtype=int)
    reached = np.zeros((grid_w, grid_h), dtype=int)

    valid = df[
        df["ball_x"].notna()
        & df["ball_i"].notna()
        & (df["minimap_valid"] == True)
    ].reset_index(drop=True)

    print(f"[xT] 유효 프레임 {len(valid)}개, 윈도우 {window_rows}행 (~{window_seconds}초)")

    if "dir_t0" in valid.columns:
        # 시작팀 시점 고정 2패스 — 점유 교대 시 좌표 미러 점프로 인한
        # 허위 '골존 도달' 오염 제거. 패스A = T0 공격 시점, 패스B = 미러.
        print("[xT] 시점 고정 2패스 모드 (dir_t0 사용)")
        gw = grid_w
        i_t0 = np.where(valid["dir_t0"] == 1,
                        valid["ball_i"], (gw - 1) - valid["ball_i"])
        for view in (i_t0, (gw - 1) - i_t0):
            arr = np.asarray(view, dtype=float)
            js = valid["ball_j_norm"].to_numpy()
            for idx in range(len(valid)):
                i, j = arr[idx], js[idx]
                if np.isnan(i) or np.isnan(j) or i >= goal_zone_i:
                    continue
                starts[int(i), int(j)] += 1
                fut = arr[idx + 1: idx + 1 + window_rows]
                if len(fut) and np.nanmax(fut) >= goal_zone_i:
                    reached[int(i), int(j)] += 1
    else:
        # 구버전 경로 (dir_t0 없는 events.csv 호환)
        for idx, row in valid.iterrows():
            i = int(row["ball_i_norm"])
            j = int(row["ball_j_norm"])
            if i >= goal_zone_i:
                continue  # 이미 골대 영역 - 시작점 안 됨
            starts[i, j] += 1

            future = valid.iloc[idx + 1 : idx + 1 + window_rows]
            future_i = future["ball_i_norm"].dropna()
            if (future_i >= goal_zone_i).any():
                reached[i, j] += 1

    if fold:
        starts = starts + starts[:, ::-1]
        reached = reached + reached[:, ::-1]

    if smooth_sigma > 0:
        s = cv2.GaussianBlur(starts.astype(np.float64), (0, 0), smooth_sigma)
        r = cv2.GaussianBlur(reached.astype(np.float64), (0, 0), smooth_sigma)
        V = np.divide(r, s, out=np.zeros_like(s), where=s > 1e-9)
    else:
        V = np.divide(reached, starts,
                       out=np.zeros_like(starts, dtype=float),
                       where=starts > 0)

    return V, starts, reached, (grid_w, grid_h)


def visualize_heatmap(V, save_path: str, scale: int = 50, show: bool = True):
    """히트맵: 가로 i (0=우리골대, max=상대골대), 세로 j."""
    grid_w, grid_h = V.shape
    img = np.zeros((grid_h * scale, grid_w * scale, 3), dtype=np.uint8)

    vmax = V.max() if V.max() > 0 else 1

    for i in range(grid_w):
        for j in range(grid_h):
            v_norm = V[i, j] / vmax  # 0~1
            r = int(v_norm * 255)
            b = 255 - r
            color = (b, 30, r)  # BGR

            x0, y0 = i * scale, j * scale
            x1, y1 = (i + 1) * scale, (j + 1) * scale
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

            text = f"{V[i, j]:.2f}"
            cv2.putText(img, text, (x0 + 4, y0 + scale - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 골대 라인 표시 (우측 끝)
    cv2.line(img, (grid_w * scale - 3, 0),
             (grid_w * scale - 3, grid_h * scale), (0, 255, 255), 5)
    cv2.putText(img, "Goal", (grid_w * scale - 70, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(save_path, img)
    print(f"[저장] {save_path}")

    if show:
        win = "xT Value Heatmap (any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(win, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(show: bool = True):
    V, starts, reached, (grid_w, grid_h) = compute_xt()

    print(f"\n[V(i,j) 통계]")
    print(f"  최댓값  : {V.max():.3f}")
    if (V > 0).any():
        print(f"  평균    : {V[V > 0].mean():.3f} (0 제외)")
    print(f"  비어있는 셀: {int((starts == 0).sum())}/{grid_w * grid_h}")
    print(f"  총 시작 : {int(starts.sum())}")
    print(f"  총 도달 : {int(reached.sum())}")

    max_idx = np.unravel_index(V.argmax(), V.shape)
    print(f"  최대 셀 : i={max_idx[0]}, j={max_idx[1]}, V={V[max_idx]:.3f}")

    visualize_heatmap(V, OUT_HEATMAP, show=show)

    out = {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "V": V.tolist(),
        "starts": starts.tolist(),
        "reached": reached.tolist(),
    }
    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    print(f"[저장] {OUT_JSON}")


if __name__ == "__main__":
    import sys
    show = "--no-display" not in sys.argv
    main(show=show)

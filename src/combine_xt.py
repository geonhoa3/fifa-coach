"""
V 통합 — 여러 영상의 xt_values.json 의 starts/reached 를 합쳐서 V_user 생성.

각 영상의 starts(i,j), reached(i,j) 행렬은 합산 가능 (위치는 정규화됨).
합산 후 V = reached / starts.

실행:
  python -m src.combine_xt                       # 기본: match04,05,06,07,08
  python -m src.combine_xt match04 match05 ...    # 영상 지정
"""

import json
import sys
import numpy as np
from pathlib import Path

from src.xt_model import visualize_heatmap


DEFAULT_VIDEOS = ["match04", "match05", "match06", "match07", "match08"]
OUT_DIR = Path("data/output/combined")


def combine(video_names: list, out_name: str = "V_user"):
    starts_sum = None
    reached_sum = None
    grid_w, grid_h = None, None

    print(f"[통합] {len(video_names)}개 영상")
    for name in video_names:
        xt_path = Path(f"data/output/{name}/xt_values.json")
        if not xt_path.exists():
            print(f"  [경고] {xt_path} 없음 → 스킵")
            continue

        data = json.loads(xt_path.read_text())
        starts = np.array(data["starts"])
        reached = np.array(data["reached"])

        if starts_sum is None:
            grid_w, grid_h = starts.shape
            starts_sum = starts.copy()
            reached_sum = reached.copy()
        else:
            starts_sum += starts
            reached_sum += reached

        print(f"  [{name}] 시작 {int(starts.sum())} / 도달 {int(reached.sum())}")

    if starts_sum is None:
        print("[실패] 합칠 데이터 없음")
        return

    # 합산 카운트에 가우시안 스무딩 후 나눔 (xt_model 과 동일 정책).
    # 영상별 json 은 이미 좌우 folding 반영 → 여기선 스무딩만.
    import cv2
    s = cv2.GaussianBlur(starts_sum.astype(np.float64), (0, 0), 0.8)
    r = cv2.GaussianBlur(reached_sum.astype(np.float64), (0, 0), 0.8)
    V = np.divide(r, s, out=np.zeros_like(s), where=s > 1e-9)

    print(f"\n[통합 V_user 통계]")
    print(f"  총 시작 : {int(starts_sum.sum())}")
    print(f"  총 도달 : {int(reached_sum.sum())}")
    print(f"  V 최댓값: {V.max():.3f}")
    if (V > 0).any():
        print(f"  V 평균  : {V[V > 0].mean():.3f} (0 제외)")
    total_cells = grid_w * grid_h
    empty = int((starts_sum == 0).sum())
    sparse = int(((starts_sum > 0) & (starts_sum < 5)).sum())
    dense = int((starts_sum >= 10).sum())
    print(f"  비어있는 셀  : {empty}/{total_cells}")
    print(f"  시도 < 5 셀  : {sparse}/{total_cells} (불안정)")
    print(f"  시도 >= 10 셀: {dense}/{total_cells} (안정)")

    # 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "grid_w": int(grid_w),
        "grid_h": int(grid_h),
        "videos": video_names,
        "V": V.tolist(),
        "starts": starts_sum.tolist(),
        "reached": reached_sum.tolist(),
    }
    (OUT_DIR / f"{out_name}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[저장] {OUT_DIR}/{out_name}.json")

    visualize_heatmap(V, str(OUT_DIR / f"{out_name}_heatmap.png"), show=False)
    print(f"[저장] {OUT_DIR}/{out_name}_heatmap.png")


def main():
    args = sys.argv[1:]
    out_name = "V_user"
    videos = []
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            out_name = args[i + 1]
            i += 2
        else:
            videos.append(args[i])
            i += 1
    if not videos:
        videos = DEFAULT_VIDEOS
    combine(videos, out_name)


if __name__ == "__main__":
    main()

"""
코칭 사이클 진척도 추적 — 처방 전 vs 처방 후 V 비교.

V_user_before (처방 전 영상들의 V)
V_user_after  (처방 후 영상들의 V)

출력:
  - 가장 개선된 셀 Top 5 (코칭 효과)
  - 가장 악화된 셀 Top 5 (후퇴 또는 노이즈)
  - V_pro 대비 부족 셀 개수 변화 (전체 코칭 진척도)
  - 3-panel 시각화 (Before / After / Progress)

실행:
  1. 처방 전 V 저장:
     python -m src.combine_xt match04 match05 match06 match07 match08 --output V_user_before

  2. 새 영상 처리 후 처방 후 V 저장:
     python -m src.combine_xt match09 match10 ... --output V_user_after

  3. 진척도 측정:
     python -m src.compare_progress
"""

import json
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BEFORE_PATH = "data/output/combined/V_user_before.json"
AFTER_PATH = "data/output/combined/V_user_after.json"
PRO_PATH = "data/output/combined/V_pro.json"
OUT_PNG = "data/output/combined/V_progress.png"
OUT_CARD = "data/output/combined/V_progress_card.png"

# 축구장 시각화 상수
PITCH_W = 1050  # 105m × 10
PITCH_H = 680   # 68m × 10
GRID_W = 16
GRID_H = 12

X_ZONES = [
    (0, 3, "우리 페널티 부근"),
    (3, 6, "우리 진영"),
    (6, 8, "우리 하프라인 근처"),
    (8, 10, "상대 하프라인 근처"),
    (10, 13, "상대 진영"),
    (13, 16, "상대 페널티 부근"),
]
Y_ZONES = [
    (0, 4, "위쪽 측면"),
    (4, 8, "중앙"),
    (8, 12, "아래쪽 측면"),
]


def cell_to_position(i: int, j: int) -> str:
    """그리드 셀 → 축구 영역 한국어 (세부)."""
    if i <= 2:
        zx = "우리 페널티에어리어"
    elif i <= 5:
        zx = "우리 진영 미드필드"
    elif i <= 7:
        zx = "하프라인 근처(우리쪽)"
    elif i <= 9:
        zx = "하프라인 근처(상대쪽)"
    elif i <= 12:
        zx = "상대 진영 미드필드"
    else:
        zx = "상대 페널티에어리어"

    if j <= 1:
        zy = "위쪽 터치라인"
    elif j <= 3:
        zy = "위쪽 측면"
    elif j <= 7:
        zy = "중앙"
    elif j <= 9:
        zy = "아래쪽 측면"
    else:
        zy = "아래쪽 터치라인"
    return f"{zx} · {zy}"


def cell_to_region(i: int, j: int) -> str:
    """그리드 셀 → 영역 (집계용, 18개)."""
    if i <= 2:
        zx = "우리 페널티 부근"
    elif i <= 5:
        zx = "우리 진영"
    elif i <= 7:
        zx = "우리 하프라인 근처"
    elif i <= 9:
        zx = "상대 하프라인 근처"
    elif i <= 12:
        zx = "상대 진영"
    else:
        zx = "상대 페널티 부근"

    if j <= 3:
        zy = "위쪽 측면"
    elif j <= 7:
        zy = "중앙"
    else:
        zy = "아래쪽 측면"
    return f"{zx} {zy}"


def aggregate_by_region(V_before, V_after, starts_before, starts_after,
                          n_videos_before: int = 1, n_videos_after: int = 1):
    """영역별 집계: 영상당 평균 시도 / 평균 V."""
    regions = {}
    grid_w, grid_h = V_before.shape
    for i in range(grid_w):
        for j in range(grid_h):
            region = cell_to_region(i, j)
            d = regions.setdefault(region, {
                "n_cells": 0,
                "v_before_sum": 0, "n_v_before": 0,
                "v_after_sum": 0, "n_v_after": 0,
                "starts_before": 0, "starts_after": 0,
            })
            d["n_cells"] += 1
            d["starts_before"] += int(starts_before[i, j])
            d["starts_after"] += int(starts_after[i, j])
            if starts_before[i, j] >= 3:
                d["v_before_sum"] += float(V_before[i, j])
                d["n_v_before"] += 1
            if starts_after[i, j] >= 3:
                d["v_after_sum"] += float(V_after[i, j])
                d["n_v_after"] += 1

    for d in regions.values():
        d["v_before_avg"] = d["v_before_sum"] / max(d["n_v_before"], 1)
        d["v_after_avg"] = d["v_after_sum"] / max(d["n_v_after"], 1)
        d["v_change"] = d["v_after_avg"] - d["v_before_avg"]
        # 영상당 평균 시도 (영상 수 차이 정규화)
        before_per_video = d["starts_before"] / n_videos_before
        after_per_video = d["starts_after"] / n_videos_after
        if before_per_video > 0:
            d["starts_change_pct"] = (after_per_video - before_per_video) / before_per_video * 100
        else:
            d["starts_change_pct"] = 0
        d["before_per_video"] = before_per_video
        d["after_per_video"] = after_per_video
    return regions


def interpret_region(region: str, d: dict) -> str:
    """
    영역 변화를 자연어 한 문장으로.
    V 변화 = 그 위치에서 시작했을 때 슛까지 도달한 비율의 변화
    starts 변화 = 그 위치에 공이 머문 빈도 변화 (영상당 평균)
    """
    v_ch = d["v_change"]
    s_pct = d["starts_change_pct"]
    n_after = d["after_per_video"]
    n_before = d["before_per_video"]

    # 데이터 너무 적으면 무시
    if n_after < 3 and n_before < 3:
        return None

    if v_ch > 0.1 and s_pct > 20:
        return (f"{region}: 더 자주 활용(+{s_pct:.0f}%) 하고 거기서 슛까지 잘 연결됨 "
                f"(슛 도달률 +{v_ch * 100:.0f}%p)")
    if v_ch > 0.1:
        return (f"{region}: 슛까지 잘 연결됨 (슛 도달률 +{v_ch * 100:.0f}%p, "
                f"방문 빈도 {s_pct:+.0f}%)")
    if v_ch < -0.1 and s_pct < -20:
        return (f"{region}: 방문 빈도 {s_pct:.0f}% 감소 + 슛 연결도 떨어짐 "
                f"(슛 도달률 {v_ch * 100:.0f}%p)")
    if v_ch < -0.1:
        return (f"{region}: 슛 연결 어려워짐 (슛 도달률 {v_ch * 100:.0f}%p, "
                f"빈도 {s_pct:+.0f}%)")
    if s_pct > 30:
        return f"{region}: 방문 빈도 +{s_pct:.0f}% 증가 (슛 도달률은 비슷)"
    if s_pct < -30:
        return (f"{region}: 방문 빈도 {s_pct:.0f}% 감소 — "
                f"이 영역 활용 자체가 줄어듦")
    return None


def load_v(path: str):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())
    V = np.array(data["V"])
    starts = np.array(data["starts"])
    n_videos = max(1, len(data.get("videos", [])))
    return V, starts, n_videos


def get_korean_font(size: int):
    """한글 지원 폰트 fallback."""
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_pitch_lines(draw: ImageDraw.ImageDraw, x0: int = 0, y0: int = 0,
                     w: int = PITCH_W, h: int = PITCH_H):
    """OpenCV 좌표계 축구장 라인."""
    draw.rectangle([x0, y0, x0 + w - 1, y0 + h - 1],
                   fill=(48, 110, 60), outline=(255, 255, 255), width=3)
    cx, cy = x0 + w // 2, y0 + h // 2
    draw.line([(cx, y0), (cx, y0 + h)], fill=(255, 255, 255), width=2)
    r = 91  # 9.15m × 10
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(255, 255, 255), width=2)
    pa_w, pa_h = 165, 403
    pa_y = y0 + (h - pa_h) // 2
    draw.rectangle([x0, pa_y, x0 + pa_w, pa_y + pa_h],
                   outline=(255, 255, 255), width=2)
    draw.rectangle([x0 + w - pa_w, pa_y, x0 + w, pa_y + pa_h],
                   outline=(255, 255, 255), width=2)
    gb_w, gb_h = 55, 183
    gb_y = y0 + (h - gb_h) // 2
    draw.rectangle([x0, gb_y, x0 + gb_w, gb_y + gb_h],
                   outline=(255, 255, 255), width=2)
    draw.rectangle([x0 + w - gb_w, gb_y, x0 + w, gb_y + gb_h],
                   outline=(255, 255, 255), width=2)


def region_center(region_full: str, offset_x: int = 0, offset_y: int = 0):
    """'상대 진영 중앙' → (x, y) 픽셀 중심."""
    for (i_s, i_e, n_x) in X_ZONES:
        if region_full.startswith(n_x):
            rest = region_full[len(n_x):].strip()
            for (j_s, j_e, n_y) in Y_ZONES:
                if rest == n_y:
                    x = offset_x + int((i_s + i_e) / 2 / GRID_W * PITCH_W)
                    y = offset_y + int((j_s + j_e) / 2 / GRID_H * PITCH_H)
                    return x, y
    return None


def render_region_panel(regions: dict, title: str) -> Image.Image:
    """A 패널: 18영역 색칠 + 슛도달률(%p) / 빈도(%) 라벨."""
    H_TITLE = 50
    img = Image.new("RGB", (PITCH_W, PITCH_H + H_TITLE), (20, 20, 20))
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = get_korean_font(28)
    draw.text((20, 10), title, font=font_title, fill=(255, 255, 255))

    draw_pitch_lines(draw, x0=0, y0=H_TITLE)

    font_main = get_korean_font(20)
    font_sub = get_korean_font(15)

    for (i_s, i_e, n_x) in X_ZONES:
        for (j_s, j_e, n_y) in Y_ZONES:
            region = f"{n_x} {n_y}"
            d = regions.get(region)
            if d is None:
                continue
            x1 = int(i_s / GRID_W * PITCH_W)
            x2 = int(i_e / GRID_W * PITCH_W)
            y1 = H_TITLE + int(j_s / GRID_H * PITCH_H)
            y2 = H_TITLE + int(j_e / GRID_H * PITCH_H)

            v_ch = d["v_change"]
            s_pct = d["starts_change_pct"]
            n_after = d["after_per_video"]
            n_before = d["before_per_video"]

            if n_after < 3 and n_before < 3:
                color = (80, 80, 80, 80)
                show_label = False
            elif v_ch > 0.05:
                alpha = min(int(abs(v_ch) * 500) + 80, 210)
                color = (40, 200, 80, alpha)
                show_label = True
            elif v_ch < -0.05:
                alpha = min(int(abs(v_ch) * 500) + 80, 210)
                color = (220, 60, 60, alpha)
                show_label = True
            else:
                color = (140, 140, 140, 60)
                show_label = True

            draw.rectangle([x1, y1, x2, y2], fill=color,
                           outline=(255, 255, 255, 110), width=1)

            if show_label:
                v_text = f"슛 {v_ch * 100:+.0f}%p"
                s_text = f"빈도 {s_pct:+.0f}%"
                draw.text((x1 + 6, y1 + 6), v_text,
                          font=font_main, fill=(255, 255, 255))
                draw.text((x1 + 6, y1 + 30), s_text,
                          font=font_sub, fill=(230, 230, 230))

    # 범례
    lg_y = H_TITLE + PITCH_H - 26
    draw.rectangle([PITCH_W - 260, lg_y, PITCH_W - 240, lg_y + 16],
                   fill=(40, 200, 80, 220))
    draw.text((PITCH_W - 235, lg_y - 3), "개선",
              font=font_sub, fill=(255, 255, 255))
    draw.rectangle([PITCH_W - 175, lg_y, PITCH_W - 155, lg_y + 16],
                   fill=(220, 60, 60, 220))
    draw.text((PITCH_W - 150, lg_y - 3), "악화",
              font=font_sub, fill=(255, 255, 255))
    draw.rectangle([PITCH_W - 90, lg_y, PITCH_W - 70, lg_y + 16],
                   fill=(80, 80, 80, 200))
    draw.text((PITCH_W - 65, lg_y - 3), "데이터 부족",
              font=font_sub, fill=(220, 220, 220))

    return img


def render_top_panel(improved_msgs: list, worsened_msgs: list) -> Image.Image:
    """C 패널: Top3 개선/악화 위치 표시 + 자막."""
    H_TITLE = 50
    CAP_H = 320
    img = Image.new("RGB", (PITCH_W, H_TITLE + PITCH_H + CAP_H), (20, 20, 20))
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = get_korean_font(28)
    draw.text((20, 10), "핵심 변화 — Top 3 개선 / Top 3 악화",
              font=font_title, fill=(255, 255, 255))

    draw_pitch_lines(draw, x0=0, y0=H_TITLE)

    font_circle = get_korean_font(30)
    font_cap_h = get_korean_font(22)
    font_cap = get_korean_font(18)

    def _draw_circle(rank, region_full, color):
        c = region_center(region_full, offset_y=H_TITLE)
        if c is None:
            return
        x, y = c
        draw.ellipse([x - 42, y - 42, x + 42, y + 42],
                     fill=color, outline=(255, 255, 255), width=3)
        draw.text((x - 11, y - 19), str(rank),
                  font=font_circle, fill=(255, 255, 255))

    for rank, (_, msg) in enumerate(improved_msgs[:3], start=1):
        region_full = msg.split(":")[0].strip()
        _draw_circle(rank, region_full, (40, 200, 80, 235))

    for rank, (_, msg) in enumerate(worsened_msgs[:3], start=1):
        region_full = msg.split(":")[0].strip()
        _draw_circle(rank, region_full, (220, 60, 60, 235))

    # 자막 영역
    cap_y = H_TITLE + PITCH_H + 14
    draw.text((20, cap_y), "▼ 개선 (녹색 원)",
              font=font_cap_h, fill=(80, 230, 110))
    cap_y += 32
    if not improved_msgs:
        draw.text((30, cap_y), "(없음)", font=font_cap, fill=(180, 180, 180))
        cap_y += 26
    for rank, (_, msg) in enumerate(improved_msgs[:3], start=1):
        draw.text((30, cap_y), f"{rank}. {msg}",
                  font=font_cap, fill=(220, 250, 220))
        cap_y += 26

    cap_y += 10
    draw.text((20, cap_y), "▼ 악화 (빨강 원)",
              font=font_cap_h, fill=(255, 110, 110))
    cap_y += 32
    if not worsened_msgs:
        draw.text((30, cap_y), "(없음)", font=font_cap, fill=(180, 180, 180))
        cap_y += 26
    for rank, (_, msg) in enumerate(worsened_msgs[:3], start=1):
        draw.text((30, cap_y), f"{rank}. {msg}",
                  font=font_cap, fill=(250, 220, 220))
        cap_y += 26

    return img


def render_coaching_card(regions: dict, improved_msgs: list,
                         worsened_msgs: list, out_path: str):
    """A + C 한 PNG로 합쳐 저장."""
    panel_a = render_region_panel(regions, title="영역별 변화 (18영역)")
    panel_c = render_top_panel(improved_msgs, worsened_msgs)

    total_h = panel_a.size[1] + panel_c.size[1] + 20
    out = Image.new("RGB", (PITCH_W, total_h), (10, 10, 10))
    out.paste(panel_a, (0, 0))
    out.paste(panel_c, (0, panel_a.size[1] + 20))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def render_value_panel(V, scale: int = 50):
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


def render_progress_panel(diff, scale: int = 50):
    """녹색 = 개선, 빨강 = 악화."""
    grid_w, grid_h = diff.shape
    img = np.zeros((grid_h * scale, grid_w * scale, 3), dtype=np.uint8)
    for i in range(grid_w):
        for j in range(grid_h):
            d = diff[i, j]
            if d > 0.05:
                g = min(int(d * 510), 255)
                color = (50, g, 50)
            elif d < -0.05:
                r = min(int(abs(d) * 510), 255)
                color = (50, 50, r)
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
    before = load_v(BEFORE_PATH)
    after = load_v(AFTER_PATH)
    pro = load_v(PRO_PATH)

    if before is None:
        print(f"[실패] {BEFORE_PATH} 없음.")
        print(f"  먼저: python -m src.combine_xt <처방 전 영상들> --output V_user_before")
        sys.exit(1)
    if after is None:
        print(f"[실패] {AFTER_PATH} 없음.")
        print(f"  먼저: python -m src.combine_xt <처방 후 영상들> --output V_user_after")
        sys.exit(1)

    V_before, starts_before, n_before = before
    V_after, starts_after, n_after = after
    if V_before.shape != V_after.shape:
        print(f"[실패] shape 불일치")
        sys.exit(1)

    diff = V_after - V_before
    grid_w, grid_h = V_before.shape

    print(f"\n[데이터]")
    print(f"  처방 전 영상 수: {n_before}개")
    print(f"  처방 후 영상 수: {n_after}개")
    print(f"  → 시도 수는 영상당 평균으로 정규화 (영상 수 차이 보정)")

    # 전체 통계
    print(f"\n[코칭 효과 — 전체]")
    b_cells = int((V_before > 0).sum())
    a_cells = int((V_after > 0).sum())
    b_avg = V_before[V_before > 0].mean() if b_cells > 0 else 0
    a_avg = V_after[V_after > 0].mean() if a_cells > 0 else 0
    print(f"  활용된 영역 수: {b_cells}곳 → {a_cells}곳"
          + (f"  ({a_cells - b_cells:+d}곳)" if a_cells != b_cells else ""))
    print(f"  활용된 영역의 평균 효율 (슛 도달률):")
    print(f"    {b_avg * 100:.1f}% → {a_avg * 100:.1f}%  "
          f"({(a_avg - b_avg) * 100:+.1f}%p)")
    improved = int((diff > 0.05).sum())
    worsened = int((diff < -0.05).sum())
    print(f"  변화 분포: 개선 {improved}곳 / 악화 {worsened}곳")

    # 영역 단위 집계 + 자연어 메시지 (영상 수 정규화)
    regions = aggregate_by_region(V_before, V_after, starts_before, starts_after,
                                    n_videos_before=n_before, n_videos_after=n_after)

    improved_msgs = []
    worsened_msgs = []
    for region, d in regions.items():
        msg = interpret_region(region, d)
        if msg is None:
            continue
        if d["v_change"] > 0.05 or (d["v_change"] >= 0 and d["starts_change_pct"] > 20):
            improved_msgs.append((d["v_change"] + d["starts_change_pct"] / 200, msg))
        else:
            worsened_msgs.append((d["v_change"] + d["starts_change_pct"] / 200, msg))

    improved_msgs.sort(reverse=True, key=lambda x: x[0])
    worsened_msgs.sort(key=lambda x: x[0])

    print(f"\n[개선된 부분]")
    if not improved_msgs:
        print("  (없음)")
    for _, msg in improved_msgs[:5]:
        print(f"  ✓ {msg}")

    print(f"\n[악화된 부분]")
    if not worsened_msgs:
        print("  (없음)")
    for _, msg in worsened_msgs[:5]:
        print(f"  ✗ {msg}")

    # V_pro 대비 진척도
    if pro:
        V_pro, _, _ = pro
        deficit_before = V_pro - V_before
        deficit_after = V_pro - V_after
        valid_mask = (starts_before >= 5) & (starts_after >= 5)
        n_lag_before = int(((deficit_before > 0.1) & valid_mask).sum())
        n_lag_after = int(((deficit_after > 0.1) & valid_mask).sum())
        total_valid = int(valid_mask.sum())
        print(f"\n[프로 대비 발전 정도]")
        print(f"  처방 전: {total_valid}개 분석 가능 영역 중 {n_lag_before}곳이 "
              f"프로 수준에 못 미침")
        print(f"  처방 후: {total_valid}개 중 {n_lag_after}곳이 프로 수준에 못 미침")
        delta = n_lag_after - n_lag_before
        if delta < 0:
            print(f"  → 개선 ✓ : {-delta}곳이 프로 수준에 도달")
        elif delta > 0:
            print(f"  → 후퇴 ⚠️ : {delta}곳이 추가로 부족해짐")
        else:
            print(f"  → 변화 없음")

        # 이전 약점 영역들의 개선량 (영역 집계)
        print(f"\n[이전 약점 영역의 개선량] (V_pro 대비 부족 영역)")
        weak_region_changes = {}
        for i in range(grid_w):
            for j in range(grid_h):
                if not valid_mask[i, j]:
                    continue
                gap_before = V_pro[i, j] - V_before[i, j]
                if gap_before <= 0.2:
                    continue
                region = cell_to_region(i, j)
                d = weak_region_changes.setdefault(region, {
                    "gap_before": [], "gap_after": []
                })
                d["gap_before"].append(gap_before)
                d["gap_after"].append(V_pro[i, j] - V_after[i, j])

        msgs = []
        for region, d in weak_region_changes.items():
            if not d["gap_before"]:
                continue
            avg_gap_before = float(np.mean(d["gap_before"]))
            avg_gap_after = float(np.mean(d["gap_after"]))
            closed = avg_gap_before - avg_gap_after
            if closed > 0.1:
                msg = (f"✓ {region}: 프로 대비 격차 {avg_gap_before:.2f} → "
                       f"{avg_gap_after:.2f} (격차 {closed:.2f} 좁혀짐)")
            elif closed < -0.1:
                msg = (f"✗ {region}: 프로 대비 격차 {avg_gap_before:.2f} → "
                       f"{avg_gap_after:.2f} (격차 {-closed:.2f} 더 벌어짐)")
            else:
                msg = f"~ {region}: 프로 대비 격차 비슷 (변화 {closed:+.2f})"
            msgs.append((closed, msg))

        msgs.sort(reverse=True)
        for _, msg in msgs[:8]:
            print(f"  {msg}")

    # 3-panel 시각화
    panel_before = add_label(render_value_panel(V_before), "V_before (before coaching)")
    panel_after = add_label(render_value_panel(V_after), "V_after (after coaching)")
    panel_diff = add_label(render_progress_panel(diff),
                            "Progress (green = improved, red = worse)")
    combo = np.vstack([panel_before, panel_after, panel_diff])

    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_PNG, combo)
    print(f"\n[저장] {OUT_PNG}  (참고용 3-panel 히트맵)")

    # 코칭 카드 (메인 시각화: A + C)
    render_coaching_card(regions, improved_msgs, worsened_msgs, OUT_CARD)
    print(f"[저장] {OUT_CARD}  ★ 메인 코칭 카드 — 이걸 보세요")


if __name__ == "__main__":
    main()

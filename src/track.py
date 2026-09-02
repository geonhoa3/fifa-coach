"""
영상 추가 누적 트래커 — 한 영상씩 넣을 때마다 자동 분석.

작동:
- 첫 영상: V_pro 대비 약점 + 공략 추천. baseline (V_user_first) 저장.
- 2번째+: V_pro 대비 약점 + V_user_first 대비 변화 + 공략 추천.

실행:
  python -m src.track match09             # 한 영상 추가
  python -m src.track match09 match10     # 여러 영상 추가
  python -m src.track --reset             # baseline 리셋 (manifest + V_first 삭제)
  python -m src.track --status            # 현재 manifest 확인

저장 위치:
  data/output/combined/
    user_videos.json       # manifest (어떤 영상이 누적됐는지)
    V_user.json            # 현재까지 누적 V
    V_user_first.json      # 첫 영상의 V (baseline, 고정)
    V_pro.json             # 기준선 (이미 있어야 함)
    coaching_card.png      # 매번 갱신되는 카드
"""

import json
import sys
import numpy as np
from datetime import date
from pathlib import Path
from PIL import Image, ImageDraw

from src.compare_progress import (
    X_ZONES, Y_ZONES, GRID_W, GRID_H, PITCH_W, PITCH_H,
    cell_to_region, get_korean_font, draw_pitch_lines, region_center,
    render_region_panel,
)


COMBINED = Path("data/output/combined")
MANIFEST = COMBINED / "user_videos.json"
V_USER = COMBINED / "V_user.json"
V_FIRST = COMBINED / "V_user_first.json"
V_PRO = COMBINED / "V_pro.json"
OUT_CARD = COMBINED / "coaching_card.png"


# ---------- manifest ----------

def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"videos": [], "added_at": []}


def save_manifest(m: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False),
                         encoding="utf-8")


# ---------- V 통합 ----------

def combine_videos(video_names: list):
    """xt_values.json 들을 합쳐 V/starts/reached 반환."""
    starts_sum = None
    reached_sum = None
    for name in video_names:
        p = Path(f"data/output/{name}/xt_values.json")
        if not p.exists():
            print(f"  [경고] {p} 없음 → 스킵")
            continue
        d = json.loads(p.read_text())
        starts = np.array(d["starts"])
        reached = np.array(d["reached"])
        if starts_sum is None:
            starts_sum = starts.copy()
            reached_sum = reached.copy()
        else:
            starts_sum += starts
            reached_sum += reached
    if starts_sum is None:
        return None
    V = np.divide(reached_sum, starts_sum,
                   out=np.zeros_like(starts_sum, dtype=float),
                   where=starts_sum > 0)
    return V, starts_sum, reached_sum


def save_v_snapshot(path: Path, V, starts, reached, videos: list):
    path.write_text(json.dumps({
        "grid_w": V.shape[0],
        "grid_h": V.shape[1],
        "videos": videos,
        "V": V.tolist(),
        "starts": starts.tolist(),
        "reached": reached.tolist(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def load_v(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return np.array(d["V"]), np.array(d["starts"]), int(d.get("grid_w", 16)), int(d.get("grid_h", 12))


# ---------- 분석: V_pro 대비 ----------

def analyze_vs_pro(V_user, V_pro, starts_user, starts_pro):
    """영역 단위로 약점 분류 (시급도 / 잠재력).

    영역의 V는 starts 가중 평균. freq는 영역 점유 비율(0~1).
    """
    total_user = float(starts_user.sum()) or 1.0

    by_region = {}
    for i in range(GRID_W):
        for j in range(GRID_H):
            region = cell_to_region(i, j)
            d = by_region.setdefault(region, {
                "v_user_w": 0.0, "v_pro_w": 0.0,
                "starts_user": 0.0, "starts_pro": 0.0,
            })
            s_u = float(starts_user[i, j])
            s_p = float(starts_pro[i, j])
            d["v_user_w"] += float(V_user[i, j]) * s_u
            d["v_pro_w"] += float(V_pro[i, j]) * s_p
            d["starts_user"] += s_u
            d["starts_pro"] += s_p

    for d in by_region.values():
        d["v_user_avg"] = d["v_user_w"] / max(d["starts_user"], 1)
        d["v_pro_avg"] = d["v_pro_w"] / max(d["starts_pro"], 1)
        d["gap"] = d["v_pro_avg"] - d["v_user_avg"]
        d["freq"] = d["starts_user"] / total_user
        # 음수 gap은 제외 (사용자가 프로보다 강한 영역)
        g_pos = max(d["gap"], 0.0)
        d["urgency"] = g_pos * d["freq"]
        d["potential"] = g_pos * (1 - d["freq"])

    # 약점만 (gap > 0.05) + 프로가 의미 있는 V 보이는 영역
    weak = {r: d for r, d in by_region.items()
            if d["gap"] > 0.05 and d["v_pro_avg"] > 0.05}
    urgent = sorted(weak.items(), key=lambda x: -x[1]["urgency"])
    potent = sorted(weak.items(), key=lambda x: -x[1]["potential"])
    return urgent, potent


def freq_label(freq: float) -> str:
    if freq >= 0.15: return "자주 활용"
    elif freq >= 0.07: return "어느 정도 활용"
    elif freq >= 0.03: return "가끔 활용"
    else: return "거의 안 씀"


def format_urgent(region: str, d: dict) -> str:
    return (f"{region}: {freq_label(d['freq'])}(전체 점유의 {d['freq'] * 100:.0f}%) "
            f"— 슛 도달률 프로 대비 -{d['gap'] * 100:.0f}%p")


def format_potent(region: str, d: dict) -> str:
    return (f"{region}: {freq_label(d['freq'])}(전체 점유의 {d['freq'] * 100:.1f}%) "
            f"인데 프로는 이 영역에서 슛 도달률 {d['v_pro_avg'] * 100:.0f}%")


# ---------- 분석: 첫 영상 대비 변화 ----------

def analyze_vs_first(V_first, V_curr, starts_first, starts_curr,
                       n_first: int = 1, n_curr: int = 1):
    """영역별 변화 (V_change + 빈도 변화)."""
    regions = {}
    for i in range(GRID_W):
        for j in range(GRID_H):
            region = cell_to_region(i, j)
            d = regions.setdefault(region, {
                "v_first_sum": 0, "n_first": 0,
                "v_curr_sum": 0, "n_curr": 0,
                "starts_first": 0, "starts_curr": 0,
            })
            d["starts_first"] += int(starts_first[i, j])
            d["starts_curr"] += int(starts_curr[i, j])
            if starts_first[i, j] >= 3:
                d["v_first_sum"] += float(V_first[i, j])
                d["n_first"] += 1
            if starts_curr[i, j] >= 3:
                d["v_curr_sum"] += float(V_curr[i, j])
                d["n_curr"] += 1
    for d in regions.values():
        d["v_first_avg"] = d["v_first_sum"] / max(d["n_first"], 1)
        d["v_curr_avg"] = d["v_curr_sum"] / max(d["n_curr"], 1)
        d["v_change"] = d["v_curr_avg"] - d["v_first_avg"]
        before_per_video = d["starts_first"] / n_first
        after_per_video = d["starts_curr"] / n_curr
        if before_per_video > 0:
            d["starts_change_pct"] = (after_per_video - before_per_video) / before_per_video * 100
        else:
            d["starts_change_pct"] = 0
        d["before_per_video"] = before_per_video
        d["after_per_video"] = after_per_video
    return regions


# compare_progress.interpret_region 과 동일한 톤
def interpret_change(region: str, d: dict):
    v_ch = d["v_change"]
    s_pct = d["starts_change_pct"]
    n_after = d["after_per_video"]
    n_before = d["before_per_video"]
    if n_after < 3 and n_before < 3:
        return None
    if v_ch > 0.1 and s_pct > 20:
        return f"{region}: 더 자주 활용(+{s_pct:.0f}%) + 슛 도달률 +{v_ch * 100:.0f}%p"
    if v_ch > 0.1:
        return f"{region}: 슛 도달률 +{v_ch * 100:.0f}%p (빈도 {s_pct:+.0f}%)"
    if v_ch < -0.1 and s_pct < -20:
        return f"{region}: 빈도 {s_pct:.0f}% + 슛 도달률 {v_ch * 100:.0f}%p"
    if v_ch < -0.1:
        return f"{region}: 슛 도달률 {v_ch * 100:.0f}%p (빈도 {s_pct:+.0f}%)"
    if s_pct > 30:
        return f"{region}: 빈도 +{s_pct:.0f}% (슛 도달률은 비슷)"
    if s_pct < -30:
        return f"{region}: 빈도 {s_pct:.0f}% (이 영역 활용 자체가 줄어듦)"
    return None


# ---------- 시각화: 단일 코칭 카드 ----------

def render_weakness_panel(urgent: list, potent: list) -> Image.Image:
    """V_pro 대비 약점 + 공략 추천 (Top3 시급/잠재 표시)."""
    H_TITLE = 50
    CAP_H = 320
    img = Image.new("RGB", (PITCH_W, H_TITLE + PITCH_H + CAP_H), (20, 20, 20))
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = get_korean_font(28)
    draw.text((20, 10), "V_pro 대비 약점 — 시급 / 잠재력",
              font=font_title, fill=(255, 255, 255))

    draw_pitch_lines(draw, x0=0, y0=H_TITLE)

    font_circle = get_korean_font(30)
    font_cap_h = get_korean_font(22)
    font_cap = get_korean_font(18)

    # 시급도 Top3 = 빨강 원
    for rank, (region, _) in enumerate(urgent[:3], start=1):
        c = region_center(region, offset_y=H_TITLE)
        if c is None:
            continue
        x, y = c
        draw.ellipse([x - 42, y - 42, x + 42, y + 42],
                     fill=(220, 60, 60, 235), outline=(255, 255, 255), width=3)
        draw.text((x - 11, y - 19), str(rank),
                  font=font_circle, fill=(255, 255, 255))

    # 잠재력 Top3 = 노랑 원 (구분)
    for rank, (region, _) in enumerate(potent[:3], start=1):
        c = region_center(region, offset_y=H_TITLE)
        if c is None:
            continue
        x, y = c
        # 잠재력 위치가 시급 위치와 겹치면 살짝 옆으로
        offset = 0 if all(region != u[0] for u in urgent[:3]) else 50
        draw.ellipse([x - 42 + offset, y - 42, x + 42 + offset, y + 42],
                     fill=(240, 200, 50, 235), outline=(255, 255, 255), width=3)
        draw.text((x - 11 + offset, y - 19), f"P{rank}",
                  font=get_korean_font(22), fill=(40, 40, 40))

    cap_y = H_TITLE + PITCH_H + 14
    draw.text((20, cap_y), "▼ 시급 (빨강 원) — 자주 가는데 약함",
              font=font_cap_h, fill=(255, 110, 110))
    cap_y += 32
    if not urgent:
        draw.text((30, cap_y), "(없음)", font=font_cap, fill=(180, 180, 180))
        cap_y += 26
    for rank, (region, d) in enumerate(urgent[:3], start=1):
        msg = format_urgent(region, d)
        draw.text((30, cap_y), f"{rank}. {msg}",
                  font=font_cap, fill=(250, 220, 220))
        cap_y += 26

    cap_y += 10
    draw.text((20, cap_y), "▼ 잠재력 (노랑 원, P1~P3) — 안 쓰는 약점 영역",
              font=font_cap_h, fill=(240, 220, 80))
    cap_y += 32
    if not potent:
        draw.text((30, cap_y), "(없음)", font=font_cap, fill=(180, 180, 180))
        cap_y += 26
    for rank, (region, d) in enumerate(potent[:3], start=1):
        msg = format_potent(region, d)
        draw.text((30, cap_y), f"P{rank}. {msg}",
                  font=font_cap, fill=(250, 240, 200))
        cap_y += 26

    return img


def render_card(urgent, potent, change_regions, n_videos: int,
                has_pro: bool = True):
    """카드 PNG 생성.
    - has_pro=False: 약점 패널 생략 (V_pro 없을 때)
    - n_videos==1: 변화 패널 생략 (첫 영상)
    """
    panels = []
    if has_pro and (urgent or potent):
        panels.append(render_weakness_panel(urgent, potent))
    if n_videos >= 2 and change_regions is not None:
        panels.append(render_region_panel(
            change_regions,
            title=f"첫 영상 대비 변화 (현재 {n_videos}경기 누적)"))

    if not panels:
        # V_pro 없고 영상 1개 → 안내 패널만
        H = 200
        out = Image.new("RGB", (PITCH_W, H), (20, 20, 20))
        draw = ImageDraw.Draw(out)
        font = get_korean_font(22)
        draw.text((20, 30), "코칭 카드 — 표시할 내용 없음",
                  font=get_korean_font(28), fill=(255, 255, 255))
        draw.text((20, 80), "• V_pro.json 없으면 약점 비교 안 됨",
                  font=font, fill=(220, 220, 220))
        draw.text((20, 115), "• 영상 1개로는 변화 분석 안 됨",
                  font=font, fill=(220, 220, 220))
        draw.text((20, 150), "→ 영상 추가하거나 V_pro 생성하세요",
                  font=font, fill=(255, 230, 100))
    elif len(panels) == 1:
        out = panels[0]
    else:
        total_h = sum(p.size[1] for p in panels) + 20 * (len(panels) - 1)
        out = Image.new("RGB", (PITCH_W, total_h), (10, 10, 10))
        y = 0
        for p in panels:
            out.paste(p, (0, y))
            y += p.size[1] + 20

    OUT_CARD.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT_CARD)


# ---------- main ----------

def cmd_status():
    m = load_manifest()
    print(f"[manifest]")
    print(f"  누적 영상 수: {len(m['videos'])}")
    if m["videos"]:
        for v, t in zip(m["videos"], m["added_at"]):
            print(f"    - {v} (추가일 {t})")
    print(f"  V_user_first 있음: {V_FIRST.exists()}")
    print(f"  V_pro 있음        : {V_PRO.exists()}")


def cmd_reset():
    for p in [MANIFEST, V_USER, V_FIRST]:
        if p.exists():
            p.unlink()
            print(f"  삭제: {p}")
    print("[reset] baseline 리셋됨. 다음 영상이 baseline이 됨.")


def add_and_analyze(new_videos: list):
    m = load_manifest()
    added = []
    for v in new_videos:
        if v in m["videos"]:
            print(f"  [skip] {v} (이미 누적됨)")
            continue
        if not Path(f"data/output/{v}/xt_values.json").exists():
            print(f"  [경고] {v} 처리 안 됨 (xt_values.json 없음) → 먼저 process_all 실행")
            continue
        m["videos"].append(v)
        m["added_at"].append(str(date.today()))
        added.append(v)
    if not added and not m["videos"]:
        print("[중단] 누적할 영상 없음")
        return
    save_manifest(m)

    # 첫 영상 baseline 처리
    is_first_run = not V_FIRST.exists()
    if is_first_run:
        first_video = m["videos"][0]
        result = combine_videos([first_video])
        if result is None:
            print(f"[중단] {first_video} 데이터 없음")
            return
        V_f, starts_f, reached_f = result
        save_v_snapshot(V_FIRST, V_f, starts_f, reached_f, [first_video])
        print(f"[baseline] V_user_first 저장 ({first_video})")

    # 현재 누적 V_user 재계산
    result = combine_videos(m["videos"])
    if result is None:
        print("[중단] 누적 데이터 없음")
        return
    V_u, starts_u, reached_u = result
    save_v_snapshot(V_USER, V_u, starts_u, reached_u, m["videos"])

    # V_pro 로드 (없으면 fallback 모드)
    has_pro = V_PRO.exists()
    V_p, starts_p = None, None
    if has_pro:
        V_p_data = load_v(V_PRO)
        V_p, starts_p, _, _ = V_p_data
        if V_u.shape != V_p.shape:
            print(f"[경고] shape 불일치: user {V_u.shape} vs pro {V_p.shape}. V_pro 비교 스킵.")
            has_pro = False
            V_p, starts_p = None, None

    n_videos = len(m["videos"])

    print(f"\n[입력] {', '.join(added) if added else '(추가 없음, 재분석)'} "
          f"({n_videos}경기 누적)")

    if not has_pro:
        print(f"\n[알림] V_pro.json 없음 — 본인 영상 누적 트래킹만 진행")
        print(f"  고수 비교 받으려면: 고수 영상 5+개 처리 후 "
              f"`python -m src.combine_xt <영상명>... --output V_pro`")

    urgent, potent = ([], [])
    if has_pro:
        # V_pro 대비 분석
        urgent, potent = analyze_vs_pro(V_u, V_p, starts_u, starts_p)
    print(f"\n[V_pro 대비 약점 — 시급도 Top3]")
    if not urgent:
        print("  (없음)")
    for rank, (region, d) in enumerate(urgent[:3], start=1):
        print(f"  {rank}. {format_urgent(region, d)}")

    print(f"\n[V_pro 대비 약점 — 잠재력 Top3]")
    if not potent:
        print("  (없음)")
    for rank, (region, d) in enumerate(potent[:3], start=1):
        print(f"  P{rank}. {format_potent(region, d)}")

    # 변화 분석 (2번째+)
    change_regions = None
    if n_videos >= 2:
        V_f_data = load_v(V_FIRST)
        V_f, starts_f, _, _ = V_f_data
        n_first = 1  # baseline은 첫 영상 하나
        change_regions = analyze_vs_first(V_f, V_u, starts_f, starts_u,
                                            n_first=n_first, n_curr=n_videos)
        improved = []
        worsened = []
        for region, d in change_regions.items():
            msg = interpret_change(region, d)
            if msg is None:
                continue
            if d["v_change"] > 0.05 or (d["v_change"] >= 0 and d["starts_change_pct"] > 20):
                improved.append((d["v_change"] + d["starts_change_pct"] / 200, msg))
            else:
                worsened.append((d["v_change"] + d["starts_change_pct"] / 200, msg))
        improved.sort(reverse=True)
        worsened.sort()

        print(f"\n[첫 영상 대비 변화 — 개선 Top5]")
        if not improved:
            print("  (없음)")
        for _, msg in improved[:5]:
            print(f"  ✓ {msg}")
        print(f"\n[첫 영상 대비 변화 — 악화 Top5]")
        if not worsened:
            print("  (없음)")
        for _, msg in worsened[:5]:
            print(f"  ✗ {msg}")
    else:
        print(f"\n[첫 영상 대비 변화] 영상 1개 — 다음 영상부터 분석 가능")

    # 공략 추천 통합
    print(f"\n[다음 공략 추천]")
    if urgent:
        r, d = urgent[0]
        print(f"  1순위 (시급): {r} — 마무리/전환 개선")
    if potent:
        r, d = potent[0]
        print(f"  2순위 (잠재): {r} — 활용 빈도 늘리기")

    # 카드 생성
    render_card(urgent, potent, change_regions, n_videos, has_pro=has_pro)
    print(f"\n[저장] {OUT_CARD}  ★ 코칭 카드")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        cmd_status()
        return
    if args[0] == "--reset":
        cmd_reset()
        return
    if args[0] == "--status":
        cmd_status()
        return
    add_and_analyze(args)


if __name__ == "__main__":
    main()

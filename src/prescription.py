"""
처방 분석기 — V_user vs V_pro 차이 가장 큰 약점 셀에 대한 구체적 처방.

분석 4가지:
  1. 선행 행동 통계 (프로가 약점 셀 진입 직전 5초 행동 분포)
  2. 진입 경로 시각화 (프로 경로 격자 PNG)
  3. 수비 형태 분석 (진입 시 dot 분포)
  4. 사용자 vs 프로 행동 비교 (같은 영역에서 누가 어떻게 다른지)

실행:
  python -m src.prescription
  python -m src.prescription --top 5   # 상위 N개 약점 셀 분석
"""

import json
import sys
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from pathlib import Path


def cell_to_position(i: int, j: int) -> str:
    """그리드 셀 (i_norm, j) → 축구 영역 한국어 명칭."""
    if i <= 2:
        zone_x = "우리 페널티에어리어"
    elif i <= 5:
        zone_x = "우리 진영 미드필드"
    elif i <= 7:
        zone_x = "하프라인 근처(우리쪽)"
    elif i <= 9:
        zone_x = "하프라인 근처(상대쪽)"
    elif i <= 12:
        zone_x = "상대 진영 미드필드"
    else:
        zone_x = "상대 페널티에어리어"

    if j <= 1:
        zone_y = "위쪽 터치라인"
    elif j <= 3:
        zone_y = "위쪽 측면"
    elif j <= 7:
        zone_y = "중앙"
    elif j <= 9:
        zone_y = "아래쪽 측면"
    else:
        zone_y = "아래쪽 터치라인"

    return f"{zone_x} · {zone_y}"


def _discover(prefix):
    return sorted(p.parent.name for p in Path("data/output").glob(f"{prefix}*/events.csv"))

USER_VIDEOS = _discover("match")
PRO_VIDEOS = _discover("pro")
USER_XT = "data/output/combined/V_user.json"
PRO_XT = "data/output/combined/V_pro.json"
OUT_DIR = Path("data/output/combined")
WINDOW_ROWS = 50  # 진입 직전 50행 (~5초)
MINI_W, MINI_H = 276, 156
GRID_W, GRID_H = 16, 12

# 실제 축구장 크기 (표준)
FIELD_W_M = 105  # 가로
FIELD_H_M = 68   # 세로
PX_TO_M_X = FIELD_W_M / MINI_W  # ≈ 0.38
PX_TO_M_Y = FIELD_H_M / MINI_H  # ≈ 0.44


def px_to_m(dx_px: float, dy_px: float) -> float:
    """미니맵 픽셀 거리 → 실제 미터."""
    dx_m = dx_px * PX_TO_M_X
    dy_m = dy_px * PX_TO_M_Y
    return float(np.hypot(dx_m, dy_m))


def classify_pass(dx_px: float, dy_px: float) -> str:
    """
    패스 벡터 → 종류 분류 (가로 이동 vs 세로 이동 비율 기반, attack_dir 무관).
    """
    dx_m = abs(dx_px) * PX_TO_M_X   # 필드 가로 (필드 길이 방향)
    dy_m = abs(dy_px) * PX_TO_M_Y   # 필드 세로 (사이드 ↔ 사이드)
    dist_m = float(np.hypot(dx_m, dy_m))

    if dist_m < 5:
        return f"원터치 패스 ({dist_m:.0f}m)"
    if dist_m >= 30:
        # 큰 세로 이동 = 사이드 스위치 (한 측면 → 반대 측면)
        if dy_m > dx_m * 1.3:
            return f"사이드 스위치 ({dist_m:.0f}m)"
        return f"롱패스 ({dist_m:.0f}m)"
    # 5~30m
    if dy_m > dx_m * 1.3:
        return f"횡패스 ({dist_m:.0f}m)"
    return f"중거리 패스 ({dist_m:.0f}m)"


def load_events_combined(video_names):
    """여러 영상의 events.csv 를 하나로 합침."""
    dfs = []
    for name in video_names:
        path = Path(f"data/output/{name}/events.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["_video"] = name
        df = df.reset_index(drop=True)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def find_target_cells(top_n: int = 3):
    """V_user vs V_pro 차이 가장 큰 셀 (사용자 부족) 반환."""
    user_data = json.loads(Path(USER_XT).read_text())
    pro_data = json.loads(Path(PRO_XT).read_text())
    V_user = np.array(user_data["V"])
    V_pro = np.array(pro_data["V"])
    starts_user = np.array(user_data["starts"])
    starts_pro = np.array(pro_data["starts"])
    diff = V_user - V_pro
    grid_w, grid_h = diff.shape

    candidates = []
    for i in range(grid_w):
        for j in range(grid_h):
            if starts_user[i, j] >= 5 and starts_pro[i, j] >= 5:
                candidates.append((diff[i, j], i, j, V_user[i, j], V_pro[i, j]))
    candidates.sort()
    return candidates[:top_n]


def find_arrivals(df, target_i, target_j):
    """이 영상 모음에서 해당 셀 도달한 행 인덱스."""
    if "ball_i_norm" not in df.columns:
        return []
    mask = (
        (df["ball_i_norm"] == target_i)
        & (df["ball_j_norm"] == target_j)
        & (df["minimap_valid"] == True)
    )
    return df[mask].index.tolist()


def analyze_actions(df, arrival_indices, window=WINDOW_ROWS):
    """도달 직전 window 행 행동/속도 + 시작점 + 마지막 패스 종류."""
    actions = Counter()
    outcomes = Counter()
    start_zones = Counter()
    last_pass_types = Counter()
    speeds = []
    distances = []  # px

    for idx in arrival_indices:
        start = max(0, idx - window)
        win_df = df.iloc[start:idx]
        for a in win_df["action"].dropna():
            actions[a] += 1
        for o in win_df["outcome"].dropna():
            if isinstance(o, str) and o:
                outcomes[o] += 1
        s = win_df["speed_px_per_frame"].dropna()
        if len(s):
            speeds.append(s.mean())

        valid = win_df[win_df["ball_x"].notna() & win_df["ball_i_norm"].notna()]
        if len(valid) >= 1:
            row_arrival = df.iloc[idx]
            if pd.notna(row_arrival["ball_x"]):
                first = valid.iloc[0]
                d = np.hypot(row_arrival["ball_x"] - first["ball_x"],
                             row_arrival["ball_y"] - first["ball_y"])
                distances.append(d)
                si = int(first["ball_i_norm"])
                sj = int(first["ball_j_norm"])
                start_zones[cell_to_position(si, sj)] += 1

                # 진입 직전 마지막 pass_or_shot 행동 → 패스 종류 분류
                passes_in_win = win_df[win_df["action"] == "pass_or_shot"].dropna(subset=["dx", "dy"])
                if len(passes_in_win):
                    last = passes_in_win.iloc[-1]
                    dx_px = float(last["dx"])
                    dy_px = float(last["dy"])
                    pass_type = classify_pass(dx_px, dy_px)
                    # 거리 부분 제외하고 종류만
                    pass_type_clean = pass_type.split(" (")[0]
                    last_pass_types[pass_type_clean] += 1

    avg_dist_px = float(np.mean(distances)) if distances else 0
    avg_dist_m = avg_dist_px * (PX_TO_M_X + PX_TO_M_Y) / 2

    return {
        "actions": dict(actions),
        "outcomes": dict(outcomes),
        "start_zones": dict(start_zones),
        "last_pass_types": dict(last_pass_types),
        "avg_speed": float(np.mean(speeds)) if speeds else 0,
        "avg_distance_px": avg_dist_px,
        "avg_distance_m": avg_dist_m,
        "n_cases": len(arrival_indices),
    }


def analyze_defense(df, arrival_indices, range_px: float = 50):
    """진입 시점 양 팀 dot 분포: 도착 위치 근처 양 팀 선수 수."""
    near_t0 = []  # 도착 근처 우리팀 (또는 같은 팀)
    near_t1 = []  # 도착 근처 상대팀
    for idx in arrival_indices:
        row = df.iloc[idx]
        if pd.isna(row.get("ball_x")) or pd.isna(row.get("ball_y")):
            continue
        bx, by = row["ball_x"], row["ball_y"]
        try:
            t0 = json.loads(row["t0_dots"]) if isinstance(row.get("t0_dots"), str) else []
            t1 = json.loads(row["t1_dots"]) if isinstance(row.get("t1_dots"), str) else []
        except Exception:
            continue
        n0 = sum(1 for x, y in t0 if np.hypot(x - bx, y - by) < range_px)
        n1 = sum(1 for x, y in t1 if np.hypot(x - bx, y - by) < range_px)
        near_t0.append(n0)
        near_t1.append(n1)
    return {
        "avg_near_t0": float(np.mean(near_t0)) if near_t0 else 0,
        "avg_near_t1": float(np.mean(near_t1)) if near_t1 else 0,
    }


def visualize_entry_paths(df, target_i, target_j, save_path):
    """진입 직전 경로 시각화."""
    scale = 4
    img_w = MINI_W * scale
    img_h = MINI_H * scale
    img = np.full((img_h, img_w, 3), (45, 90, 45), dtype=np.uint8)

    cell_w = img_w / GRID_W
    cell_h = img_h / GRID_H

    # 그리드 라인
    for i in range(GRID_W + 1):
        x = int(i * cell_w)
        cv2.line(img, (x, 0), (x, img_h), (70, 120, 70), 1)
    for j in range(GRID_H + 1):
        y = int(j * cell_h)
        cv2.line(img, (0, y), (img_w, y), (70, 120, 70), 1)

    # 타겟 셀 강조
    tx0, ty0 = int(target_i * cell_w), int(target_j * cell_h)
    tx1, ty1 = int((target_i + 1) * cell_w), int((target_j + 1) * cell_h)
    cv2.rectangle(img, (tx0, ty0), (tx1, ty1), (0, 255, 255), 3)
    cv2.putText(img, f"TARGET ({target_i},{target_j})",
                (tx0, max(20, ty0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    arrivals = find_arrivals(df, target_i, target_j)

    # 경로 화살표 (반투명 누적)
    overlay = img.copy()
    drawn = 0
    for idx in arrivals[:40]:
        start = max(0, idx - WINDOW_ROWS)
        win_df = df.iloc[start:idx + 1]
        valid = win_df[win_df["ball_x"].notna()]
        if len(valid) < 2:
            continue
        first = valid.iloc[0]
        last = valid.iloc[-1]
        sx, sy = int(first["ball_x"] * scale), int(first["ball_y"] * scale)
        ex, ey = int(last["ball_x"] * scale), int(last["ball_y"] * scale)
        cv2.arrowedLine(overlay, (sx, sy), (ex, ey),
                        (0, 255, 100), 2, tipLength=0.1)
        drawn += 1

    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, f"Pro entry paths: {drawn}/{len(arrivals)} cases",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"  [저장] {save_path}")


def print_stats_table(label, stats):
    print(f"\n  [{label}] ({stats['n_cases']}회)")
    if stats['n_cases'] == 0:
        print(f"    데이터 없음 — 도달 자체가 거의 없음")
        return
    total = sum(stats['actions'].values()) or 1
    print(f"    선행 5초 행동:")
    for a, c in sorted(stats['actions'].items(), key=lambda x: -x[1])[:5]:
        pct = c * 100 // total
        print(f"      {a:18s}: {c:4d} ({pct}%)")
    if stats['outcomes']:
        top_outcomes = sorted(stats['outcomes'].items(), key=lambda x: -x[1])[:3]
        outcome_str = ", ".join(f"{o}={c}" for o, c in top_outcomes)
        print(f"    주요 outcome: {outcome_str}")
    print(f"    평균 공속도: {stats['avg_speed']:.2f} px/frame")
    print(f"    평균 진입거리: {stats['avg_distance_m']:.1f}m ({stats['avg_distance_px']:.0f}px)")
    if stats.get('start_zones'):
        print(f"    시작 영역 Top 3 (5초 전 공 위치):")
        for zone, c in sorted(stats['start_zones'].items(), key=lambda x: -x[1])[:3]:
            pct = c * 100 // stats['n_cases']
            print(f"      {zone:40s}: {c}회 ({pct}%)")
    if stats.get('last_pass_types'):
        print(f"    진입 직전 마지막 패스 종류 Top 3:")
        total_pass = sum(stats['last_pass_types'].values()) or 1
        for ptype, c in sorted(stats['last_pass_types'].items(), key=lambda x: -x[1])[:3]:
            pct = c * 100 // total_pass
            print(f"      {ptype:25s}: {c}회 ({pct}%)")


def main():
    args = sys.argv[1:]
    top_n = 3
    for i, a in enumerate(args):
        if a == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])

    print(f"[처방 분석기]")
    print(f"  사용자 영상: {USER_VIDEOS}")
    print(f"  프로 영상  : {PRO_VIDEOS}")
    print(f"  타겟 셀 Top {top_n}\n")

    targets = find_target_cells(top_n=top_n)
    if not targets:
        print("[실패] 타겟 셀 없음. V_user, V_pro 둘 다 있는지 확인하세요.")
        return

    pro_df = load_events_combined(PRO_VIDEOS)
    user_df = load_events_combined(USER_VIDEOS)
    print(f"[데이터] 프로 events {len(pro_df)}행, 사용자 events {len(user_df)}행")

    summary = []
    for diff, i, j, u_v, p_v in targets:
        position = cell_to_position(i, j)
        print(f"\n{'=' * 70}")
        print(f"[타겟 #{len(summary) + 1}] {position}  (셀 {i},{j})")
        print(f"  사용자 V={u_v:.2f} vs 프로 V={p_v:.2f}  (차이 {diff:+.2f})")
        print('=' * 70)

        # 프로 분석
        pro_arrivals = find_arrivals(pro_df, i, j)
        pro_stats = analyze_actions(pro_df, pro_arrivals)
        pro_def = analyze_defense(pro_df, pro_arrivals)
        print_stats_table("프로", pro_stats)
        print(f"    진입 시 근처 dot (50px): 같은팀 {pro_def['avg_near_t0']:.1f}, "
              f"상대팀 {pro_def['avg_near_t1']:.1f}")

        # 사용자 분석
        user_arrivals = find_arrivals(user_df, i, j)
        user_stats = analyze_actions(user_df, user_arrivals)
        user_def = analyze_defense(user_df, user_arrivals)
        print_stats_table("사용자", user_stats)
        if user_stats['n_cases']:
            print(f"    진입 시 근처 dot (50px): 같은팀 {user_def['avg_near_t0']:.1f}, "
                  f"상대팀 {user_def['avg_near_t1']:.1f}")

        # 시각화
        out_path = OUT_DIR / f"prescription_({i},{j}).png"
        visualize_entry_paths(pro_df, i, j, str(out_path))

        # 처방 메시지
        print(f"\n  [처방]")
        if pro_stats['n_cases'] == 0:
            print(f"    프로도 「{position}」 도달 거의 없음 → 약점 셀로 의미 없음")
        elif user_stats['n_cases'] == 0:
            top_action = max(pro_stats['actions'], key=pro_stats['actions'].get) \
                if pro_stats['actions'] else "?"
            top_start = max(pro_stats['start_zones'], key=pro_stats['start_zones'].get) \
                if pro_stats['start_zones'] else "?"
            print(f"    너는 「{position}」 도달 자체가 거의 없음.")
            print(f"    프로는 {pro_stats['n_cases']}회 도달, 평균 진입거리 {pro_stats['avg_distance']:.0f}px.")
            print(f"    프로는 주로 「{top_start}」 에서 출발 → 「{position}」 로 진입.")
            print(f"    → 「{top_start}」 에서 공 잡으면 「{position}」 쪽으로 패스/침투 시도해라")
        else:
            diff_speed = pro_stats['avg_speed'] - user_stats['avg_speed']
            if abs(diff_speed) > 0.1:
                faster = "프로" if diff_speed > 0 else "너"
                print(f"    {faster}가 진입 시 평균 속도 {abs(diff_speed):.2f} 더 빠름")
            if pro_def['avg_near_t1'] < user_def['avg_near_t1']:
                print(f"    프로 진입 시 근처 상대 {pro_def['avg_near_t1']:.1f}명 vs "
                      f"너 {user_def['avg_near_t1']:.1f}명")
                print(f"    → 프로는 빈 공간에서 진입. 너는 압박받는 상황")
            if pro_stats['start_zones']:
                top_start = max(pro_stats['start_zones'], key=pro_stats['start_zones'].get)
                print(f"    프로 진입 시작 영역 Top 1: 「{top_start}」 → 거기서 사이드 스위치 시도")

        summary.append((i, j, pro_stats['n_cases'], user_stats['n_cases']))

    print(f"\n{'=' * 70}")
    print("[요약]")
    for i, j, p_n, u_n in summary:
        print(f"  ({i}, {j}): 프로 {p_n}회 vs 사용자 {u_n}회")
    print(f"\n[처방 시각화] {OUT_DIR}/prescription_*.png")


if __name__ == "__main__":
    main()

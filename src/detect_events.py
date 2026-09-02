"""
Step 4b — 패스 성공/실패 판정

알고리즘:
  1. pass_or_shot 시점의 공 위치 = 패스 도착 지점
  2. 도착 지점에서 가장 가까운 dot 찾기 (max_dist 이내)
  3. 그 dot의 팀 → pass_success / pass_intercepted / uncertain
  4. 우리팀은 조작 선수(빨강 dot) 위치 분포로 자동 추정

실행:
  python -m src.detect_events
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


DEFAULT_CSV_IN = "data/output/motion.csv"
DEFAULT_CSV_OUT = "data/output/events.csv"

# 스코어 배너 왼쪽/오른쪽 팀명에서 우리 편을 식별하는 닉네임 앵커
# (FC 온라인 1:1 은 유저가 항상 왼쪽이지만, OCR 팀명으로 재확인)
USER_NICK = "09이건호"


def _parse_dots(s):
    if not isinstance(s, str) or not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []


def closest_dot_team(ball_xy, t0_dots, t1_dots, max_dist: float = 40):
    """
    공 위치에서 T0 최단거리 vs T1 최단거리 비교 → 더 가까운 팀 반환.
    둘 다 max_dist 초과면 None (도착점 근처에 아무도 없음 = uncertain).
    """
    bx, by = ball_xy

    def min_dist(dots):
        if not dots:
            return float("inf")
        return min(np.hypot(dx - bx, dy - by) for dx, dy in dots)

    d0 = min_dist(t0_dots)
    d1 = min_dist(t1_dots)

    if min(d0, d1) > max_dist:
        return None
    if d0 == d1 == float("inf"):
        return None
    return 0 if d0 < d1 else 1


def detect_our_team(df: pd.DataFrame, max_dist: float = 30) -> int:
    """
    조작 선수(player_x, player_y)는 빨강 dot.
    그 위치 근처 dot이 어느 팀에 더 많이 매칭되는지로 우리팀 판정.
    """
    sub = df[df["player_x"].notna() & df["player_y"].notna()]
    t0_hits, t1_hits = 0, 0
    for _, row in sub.iterrows():
        t0 = _parse_dots(row.get("t0_dots", ""))
        t1 = _parse_dots(row.get("t1_dots", ""))
        team = closest_dot_team((row["player_x"], row["player_y"]), t0, t1, max_dist=max_dist)
        if team == 0:
            t0_hits += 1
        elif team == 1:
            t1_hits += 1
    print(f"[우리팀 추정] 조작선수 근처 T0={t0_hits}, T1={t1_hits}")
    return 0 if t0_hits >= t1_hits else 1


def detect_red_uniform_case(df: pd.DataFrame, threshold: float = 0.5,
                             near_px: float = 10.0,
                             near_frac_threshold: float = 0.10,
                             min_samples: int = 100) -> bool:
    """
    우리팀이 빨강 유니폼인지 자동 감지 (v2 — 공↔빨강 dot 거리 기반).

    빨강 유니폼 케이스: 우리 공 소유 시에도 조작선수 dot이 빨강으로 보임
      → 볼 캐리어가 빨강 dot = 공 마커와 빨강 dot이 자주 겹침 (dist < near_px)
    정상 케이스: 우리 소유 시 빨강 dot이 공 마커에 가려져 안 보임
      → 빨강이 보일 땐 수비 추격 상황 = 공과 거의 안 겹침

    구버전(검출률 > 50%)은 정상 케이스 검출률이 42~61%로 널뛰어 오판 빈발
    (match08: 65%로 빨강 유니폼 오판 → 소유권 전체 반전 → 자기 골 3개를
    상대 골로 판정). 실측: 정상 케이스 9경기 모두 dist<10px 비율 ≤ 0.4%.
    """
    valid = df[df["minimap_valid"] == True] if "minimap_valid" in df.columns else df
    if len(valid) == 0:
        return False
    red_rate = valid["player_x"].notna().sum() / len(valid)

    both = valid[valid["ball_x"].notna() & valid["player_x"].notna()]
    if len(both) >= min_samples:
        d = np.hypot(both["ball_x"] - both["player_x"],
                     both["ball_y"] - both["player_y"])
        near_frac = float((d < near_px).mean())
        red_uniform = near_frac >= near_frac_threshold
        print(f"[유니폼 케이스 판별] 빨강 dot 검출률 {red_rate*100:.0f}%, "
              f"공↔빨강 겹침(<{near_px:.0f}px) {near_frac*100:.1f}% "
              f"→ {'빨강 유니폼' if red_uniform else '정상'} 케이스")
        return red_uniform

    # 표본 부족 → 구버전 검출률 휴리스틱 폴백
    print(f"[유니폼 케이스 판별] 동시 검출 표본 부족({len(both)}) → "
          f"검출률 휴리스틱 폴백")
    print(f"[빨강 dot 검출률] {red_rate * 100:.0f}%  "
          f"({'빨강 유니폼' if red_rate > threshold else '정상'} 케이스)")
    if abs(red_rate - threshold) < 0.10:
        print(f"[경고] 검출률이 임계값에 근접 — 유니폼 케이스 오판 위험. "
              f"점유율/패스 통계가 뒤집혀 보이면 케이스 반전을 의심할 것.")
    return red_rate > threshold


def is_our_possession(row, red_uniform: bool = False) -> bool:
    """
    우리팀 공 소유 판정.

    정상 케이스: 빨간 dot 검출 안 됨 (공에 가려짐) = 우리 공 소유
    빨강 유니폼 케이스: 빨간 dot 검출됨 (조작 선수 dot이 빨강) = 우리 공 소유
    """
    if "minimap_valid" in row and not row["minimap_valid"]:
        return False
    if pd.isna(row["ball_x"]) or pd.isna(row["ball_y"]):
        return False
    has_red = pd.notna(row["player_x"]) and pd.notna(row["player_y"])
    if red_uniform:
        return has_red       # 빨강 보임 = 우리 공 소유
    return not has_red       # 정상: 빨강 안 보임 = 우리 공 소유


def determine_possession_team(row, max_dist: float = 20):
    """공 위치 → 가장 가까운 dot 의 팀 (0 or 1). observer 모드용.

    조작 선수 정보(player_x) 와 무관하게 공 근처 dot 으로만 판정.
    양팀 모두 색 dot 가지는 관전 모드 영상에서 사용.
    """
    if pd.isna(row.get("ball_x")) or pd.isna(row.get("ball_y")):
        return None
    if "minimap_valid" in row and not row["minimap_valid"]:
        return None
    t0 = _parse_dots(row.get("t0_dots", ""))
    t1 = _parse_dots(row.get("t1_dots", ""))
    return closest_dot_team((row["ball_x"], row["ball_y"]), t0, t1, max_dist=max_dist)


def classify_outcomes_observer(df: pd.DataFrame,
                                window_before: int = 3,
                                window_after: int = 10,
                                goal_left: float = 15,
                                goal_right: float = 261) -> pd.DataFrame:
    """관전 모드: 양팀 모두 분석. team 컬럼 추가 (0 or 1).

    기본 모드와 차이:
      - "우리팀" 개념 없음. 매 pass_or_shot 의 점유 팀을 공-근처 dot 으로 추정.
      - 양팀 시점 모두 events.csv 에 들어감.
      - shot 판정: 양쪽 골대 영역 어디로 가든 shot (어느 팀의 슛인지는 team 으로 구분).
    """
    df["outcome"] = ""
    df["team"] = pd.NA

    # 매 프레임 점유 팀 마킹 — pos_team 으로 보존 (grid.py 정규화용)
    df["pos_team"] = df.apply(determine_possession_team, axis=1)
    df["_pos_team"] = df["pos_team"]

    mask = (df["action"] == "pass_or_shot") & df["ball_x"].notna() & df["ball_y"].notna()
    pass_indices = df[mask].index.tolist()

    for idx in pass_indices:
        # 직전 N프레임 + 현재의 점유 팀 → 최빈값으로 결정
        past = df.loc[max(0, idx - window_before): idx]
        past_teams = past["_pos_team"].dropna()
        if len(past_teams) == 0:
            df.at[idx, "outcome"] = "uncertain"
            continue
        team = int(past_teams.mode().iloc[0])
        df.at[idx, "team"] = team

        bx = df.at[idx, "ball_x"]
        # 양쪽 골대 영역 → shot
        if bx < goal_left or bx > goal_right:
            df.at[idx, "outcome"] = "shot"
            continue

        # 직후 윈도우 안에 같은 팀 점유 회복?
        future = df.loc[idx + 1: idx + window_after]
        future_teams = future["_pos_team"].dropna()
        if len(future_teams) == 0:
            df.at[idx, "outcome"] = "uncertain"
            continue

        if (future_teams == team).any():
            df.at[idx, "outcome"] = "pass_success"
        else:
            df.at[idx, "outcome"] = "pass_intercepted"

    df.drop(columns=["_pos_team"], inplace=True)

    # 통계
    if "team" in df.columns:
        events = df[df["action"] == "pass_or_shot"]
        team_dist = events["team"].value_counts(dropna=False)
        print(f"[관전 모드] 팀별 이벤트 분포:")
        print(team_dist.to_string())
    return df


def invalidate_teleports(df: pd.DataFrame, teleport_speed: float = 8.0) -> pd.DataFrame:
    """순간이동 아티팩트 무효화.

    골 직후 킥오프 리셋·리플레이 전환 시 공 좌표가 한 샘플 사이에 크게 점프
    → 초고속 이동으로 보여 pass_or_shot 으로 오분류됨.
    실측 경기 최고 속도 ~3.5 px/frame, 순간이동은 19~25 px/frame.
    teleport_speed(=8.0) 이상이면 outcome='teleport' 로 표시 (평가 제외).
    """
    if "speed_px_per_frame" not in df.columns:
        return df
    mask = ((df["action"] == "pass_or_shot")
            & (df["speed_px_per_frame"] >= teleport_speed))
    n = int(mask.sum())
    if n:
        df.loc[mask, "outcome"] = "teleport"
        print(f"[순간이동 필터] {n}개 이벤트 무효화 (speed >= {teleport_speed} px/frame)")
    return df


def merge_adjacent_shots(df: pd.DataFrame, min_gap_frames: int = 30) -> pd.DataFrame:
    """
    같은 슛이 여러 프레임에서 검출되는 경우 첫 번째만 유지.
    min_gap_frames(=30 = 1초) 이내 추가 shot 은 'shot_dup' 으로 표시.
    """
    shot_indices = df[df["outcome"] == "shot"].index.tolist()
    if len(shot_indices) <= 1:
        return df

    last_shot_frame = df.at[shot_indices[0], "frame"]
    for i in range(1, len(shot_indices)):
        idx = shot_indices[i]
        cur_frame = df.at[idx, "frame"]
        if cur_frame - last_shot_frame < min_gap_frames:
            df.at[idx, "outcome"] = "shot_dup"
        else:
            last_shot_frame = cur_frame
    return df


def infer_goals(df: pd.DataFrame, mini_w: int,
                red_uniform: bool = False,
                observer: bool = False,
                edge_frac: float = 0.15,
                min_gap_sec: float = 2.0,
                flight_window_sec: float = 1.2,
                min_flight_rows: int = 2,
                lookback_sec: float = 6.0,
                garbage_sec: float = 1.5,
                cooldown_sec: float = 10.0) -> pd.DataFrame:
    """골 추론 — 미니맵 시그니처 기반 (match08 3골 전부 놓친 사고의 재발 방지).

    골의 미니맵 시그니처:
      1. 공이 골대 끝단(edge_frac 이내)에서 소실
      2. 소실 직전 flight_window_sec 안에 pass_or_shot 프레임 다수 (슛 비행)
      3. 이후 min_gap_sec 이상 공 미검출 (골 리플레이/세리머니)

    슛 판정이 '공의 마지막 검출 좌표 > 골대 임계'에 의존하면, 빠른 슛일수록
    공이 골문에 박히기 전에 검출이 끊겨 골을 pass_intercepted 로 오분류한다
    (match08: x=358 소실, 임계 369 — 11px 차이로 골 3개 전부 누락).

    한계: 골대 근처 아웃 + 리플레이(선방 장면 등)도 골로 잡힐 수 있음 (PoC 수준).
    확실한 검출은 스코어보드 OCR (로드맵) 필요.

    소유권: 소실 전 [lookback, garbage] 구간의 점유 다수결.
    (소실 직전 garbage_sec 는 리플레이 좌표 오염 가능성이 있어 제외)
    → 우리 골 'goal' / 상대 골 'opponent_goal'
    슛 비행 조각(직전 pass_* 오분류 행)은 'goal_flight' 로 재라벨 → 패스 통계 오염 제거.
    """
    if "outcome" not in df.columns:
        return df

    if observer:
        poss = df.get("pos_team")
    else:
        poss = df.apply(lambda r: is_our_possession(r, red_uniform=red_uniform), axis=1)

    noball = df["ball_x"].isna()
    edge_lo, edge_hi = edge_frac * mini_w, (1 - edge_frac) * mini_w

    goals = []
    last_goal_t = -1e9
    run_start = None
    idx_list = df.index.tolist()

    def check_run(run_start_k: int, gap_end_t: float):
        """공 미검출 run 하나를 골 후보로 검사. 골이면 라벨링."""
        nonlocal last_goal_t
        gap_dur = gap_end_t - df.at[idx_list[run_start_k], "time_sec"]
        prev_k = run_start_k - 1
        if gap_dur < min_gap_sec or prev_k < 0:
            return
        prev_idx = idx_list[prev_k]
        bx = df.at[prev_idx, "ball_x"]
        t_vanish = df.at[prev_idx, "time_sec"]
        if pd.isna(bx) or (edge_lo <= bx <= edge_hi):
            return
        if t_vanish - last_goal_t < cooldown_sec:
            return
        # 슛 비행 확인 — 소실 직전에 빠른 공 이동이 실제로 있었나
        flight = df[(df["time_sec"] > t_vanish - flight_window_sec)
                    & (df["time_sec"] <= t_vanish)]
        flight_rows = flight[flight["action"] == "pass_or_shot"]
        if len(flight_rows) < min_flight_rows:
            return

        # 소유권 다수결 — 판정 가능한 행(valid + 공 검출)만 집계
        win = df[(df["time_sec"] >= t_vanish - lookback_sec)
                 & (df["time_sec"] <= t_vanish - garbage_sec)]
        win = win[(win["minimap_valid"] == True) & win["ball_x"].notna()]
        if observer:
            teams = win["pos_team"].dropna() if poss is not None else pd.Series(dtype=float)
            team = int(teams.mode().iloc[0]) if len(teams) else None
            df.at[prev_idx, "outcome"] = "goal"
            if "team" in df.columns and team is not None:
                df.at[prev_idx, "team"] = team
        else:
            our_frac = poss.loc[win.index].mean() if len(win) else 0.0
            outcome = "goal" if our_frac >= 0.5 else "opponent_goal"
            df.at[prev_idx, "outcome"] = outcome

        # 슛 비행 조각 재라벨 (골 행 제외) — pass_success/intercepted 오염 제거
        for fi in flight_rows.index:
            if fi != prev_idx and df.at[fi, "outcome"] in (
                    "pass_success", "pass_intercepted", "teleport", "shot", "shot_dup"):
                df.at[fi, "outcome"] = "goal_flight"

        last_goal_t = t_vanish
        goals.append((t_vanish, df.at[prev_idx, "outcome"], bx, gap_dur))

    for k, idx in enumerate(idx_list):
        if noball.loc[idx]:
            if run_start is None:
                run_start = k
            continue
        if run_start is None:
            continue
        # 공 미검출 run 종료 → run 검사
        check_run(run_start, gap_end_t=df.at[idx, "time_sec"])
        run_start = None

    # 영상이 공 미검출 상태로 끝나는 run — 골 세리머니/통계 화면으로 끝나는
    # 경우가 흔함 (match08: 마지막 골이 이 케이스라 누락됐었음)
    if run_start is not None:
        check_run(run_start, gap_end_t=df.at[idx_list[-1], "time_sec"])

    if goals:
        print(f"[골 추론] {len(goals)}개 검출:")
        for t, oc, bx, gap in goals:
            print(f"  {int(t//60)}:{int(t%60):02d} ({t:.1f}s)  {oc}  "
                  f"소실 x={bx:.0f}, 공백 {gap:.1f}s")
    else:
        print("[골 추론] 검출된 골 없음")
    return df


def reconcile_ocr_goals(df: pd.DataFrame, ocr_json_path,
                         margin: float = 6.0) -> pd.DataFrame:
    """goals_ocr.json (스코어보드 OCR) 을 ground truth 로 휴리스틱 골과 대조.

    - side→우리팀 매핑: 매칭된 페어 다수결 (페어 없으면 left=우리 가정 —
      FC 온라인은 유저 구단이 배너 왼쪽)
    - 매칭된 휴리스틱 골: outcome 을 OCR side 기준으로 확정
    - OCR 에 없는 휴리스틱 골: 슛으로 강등 (선방+리플레이 오탐이었을 가능성)
    - 휴리스틱이 못 잡은 OCR 골: 윈도우 내 마지막 공 검출 행에 goal 라벨
    """
    ocr_json_path = Path(ocr_json_path)
    if not ocr_json_path.exists():
        return df
    ocr = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    ocr_goals = ocr.get("goals", [])
    print(f"[OCR 대조] {ocr_json_path} — OCR 골 {len(ocr_goals)}개, "
          f"최종 스코어 {ocr.get('final_score')}")

    inferred_idx = df[df["outcome"].isin(["goal", "opponent_goal"])].index.tolist()
    used = set()
    pairs = []  # (ocr_goal, inferred_idx or None)
    for g in ocr_goals:
        lo, hi = g["t_before"] - margin, g["t_after"] + margin
        match = None
        for idx in inferred_idx:
            if idx in used:
                continue
            if lo <= df.at[idx, "time_sec"] <= hi:
                match = idx
                break
        if match is not None:
            used.add(match)
        pairs.append((g, match))

    # side→our 매핑 — 1순위: 배너 팀명 닉네임 앵커 (확실),
    # 2순위: 매칭 페어 다수결 (소유권 오판 시 통째로 반전 위험 — match13 사고)
    left_is_our = None
    names = ocr.get("team_names") or {}
    if USER_NICK:
        if USER_NICK in (names.get("left") or ""):
            left_is_our = True
        elif USER_NICK in (names.get("right") or ""):
            left_is_our = False
        if left_is_our is not None:
            print(f"  side 매핑: left={'우리' if left_is_our else '상대'} "
                  f"(닉네임 앵커: {names})")
    if left_is_our is None:
        vote_left_our = 0
        vote_left_opp = 0
        for g, idx in pairs:
            if idx is None:
                continue
            inferred_ours = df.at[idx, "outcome"] == "goal"
            if (g["side"] == "left") == inferred_ours:
                vote_left_our += 1
            else:
                vote_left_opp += 1
        left_is_our = vote_left_our >= vote_left_opp
        if vote_left_our or vote_left_opp:
            print(f"  side 매핑: left={'우리' if left_is_our else '상대'} "
                  f"(다수결 {vote_left_our}:{vote_left_opp})")

    def side_outcome(side):
        ours = (side == "left") == left_is_our
        return "goal" if ours else "opponent_goal"

    for g, idx in pairs:
        oc = side_outcome(g["side"])
        if idx is not None:
            old = df.at[idx, "outcome"]
            df.at[idx, "outcome"] = oc
            t = df.at[idx, "time_sec"]
            flag = "" if old == oc else f" (휴리스틱 {old} → 정정)"
            print(f"  {int(t//60)}:{int(t%60):02d} {oc}{flag}")
        else:
            # 휴리스틱 미검출 골 → 윈도우 내 마지막 공 검출 행에 라벨
            win = df[(df["time_sec"] >= g["t_before"] - margin)
                     & (df["time_sec"] <= g["t_after"] + margin)
                     & df["ball_x"].notna()]
            if len(win) == 0:
                win = df[(df["time_sec"] <= g["t_after"]) & df["ball_x"].notna()]
            if len(win) == 0:
                print(f"  [경고] OCR 골({g['t_after']:.0f}s) 근처에 공 검출 행이 "
                      f"없어 라벨 불가")
                continue
            idx2 = win.index[-1]
            df.at[idx2, "outcome"] = oc
            t = df.at[idx2, "time_sec"]
            print(f"  {int(t//60)}:{int(t%60):02d} {oc} (휴리스틱 미검출 — OCR 로 추가)")

    # OCR 에 없는 휴리스틱 골 → 슛으로 강등
    for idx in inferred_idx:
        if idx in used:
            continue
        old = df.at[idx, "outcome"]
        df.at[idx, "outcome"] = "shot" if old == "goal" else "opponent_shot"
        t = df.at[idx, "time_sec"]
        print(f"  {int(t//60)}:{int(t%60):02d} {old} → "
              f"{df.at[idx, 'outcome']} 강등 (OCR 스코어 변화 없음)")
    return df


def classify_outcomes(df: pd.DataFrame, our_team: int = 0,
                       window_before: int = 3,
                       window_after: int = 10,
                       goal_left: float = 15, goal_right: float = 261,
                       red_uniform: bool = None) -> pd.DataFrame:
    """
    pass_or_shot 시점 판정:
      1. 직전 window_before 행에서 우리팀 공 소유 아니었으면 → 'opponent_pass' (분석 제외)
      2. 도착점이 골대 영역 → 'shot'
      3. 직후 window_after 행 안에 our_possession 회복 → 'pass_success'
      4. 회복 안 됨 → 'pass_intercepted'
    """
    df["outcome"] = ""

    # 빨강 유니폼 자동 감지 → 로직 반전 (main에서 미리 계산했으면 재사용)
    if red_uniform is None:
        red_uniform = detect_red_uniform_case(df)
    df["_our_possession"] = df.apply(
        lambda row: is_our_possession(row, red_uniform=red_uniform), axis=1
    )

    mask = (df["action"] == "pass_or_shot") & df["ball_x"].notna() & df["ball_y"].notna()
    pass_indices = df[mask].index.tolist()

    for idx in pass_indices:
        # 1. 직전에 우리 공이었나?
        past = df.loc[max(0, idx - window_before) : idx - 1]
        was_ours = past["_our_possession"].any() if len(past) > 0 else True
        if not was_ours:
            df.at[idx, "outcome"] = "opponent_pass"
            continue

        bx = df.at[idx, "ball_x"]
        # 2. 골대 영역 → 슛 (우리팀 슛)
        if bx < goal_left or bx > goal_right:
            df.at[idx, "outcome"] = "shot"
            continue

        # 3-4. 직후 우리팀 소유 회복?
        future = df.loc[idx + 1 : idx + window_after]
        if len(future) == 0:
            df.at[idx, "outcome"] = "uncertain"
            continue

        if future["_our_possession"].any():
            df.at[idx, "outcome"] = "pass_success"
        else:
            df.at[idx, "outcome"] = "pass_intercepted"

    df.drop(columns=["_our_possession"], inplace=True)
    return df


def main(csv_in: str = DEFAULT_CSV_IN, csv_out: str = DEFAULT_CSV_OUT,
         observer: bool = False):
    print(f"[detect_events] in : {csv_in}")
    print(f"[detect_events] out: {csv_out}")
    print(f"[모드] {'관전 모드 (양팀 분석)' if observer else '기본 (한 팀 시점)'}")
    df = pd.read_csv(csv_in)

    # 슛 판정 골대 임계를 실제 미니맵 폭에 맞게 스케일
    # (기존: 276px 기준 15/261 하드코딩 → 다른 크기 미니맵에서 오분류)
    mini_w = 276
    gs_meta = Path(csv_in).parent / "game_state.meta.json"
    if gs_meta.exists():
        try:
            roi = json.loads(gs_meta.read_text()).get("roi")
            if roi:
                mini_w = roi[2]
        except Exception:
            pass
    gl, gr = mini_w * 15 / 276, mini_w * 261 / 276
    print(f"[골대 임계] 미니맵 폭 {mini_w} → left {gl:.0f}, right {gr:.0f}")

    red_uniform = False
    if observer:
        df = classify_outcomes_observer(df, goal_left=gl, goal_right=gr)
    else:
        red_uniform = detect_red_uniform_case(df)
        df = classify_outcomes(df, goal_left=gl, goal_right=gr,
                               red_uniform=red_uniform)

    # 순간이동 무효화 → 인접 슛 병합 → 골 추론
    df = invalidate_teleports(df)
    df = merge_adjacent_shots(df, min_gap_frames=30)
    df = infer_goals(df, mini_w=mini_w, red_uniform=red_uniform,
                     observer=observer)
    # 스코어보드 OCR 결과 있으면 ground truth 로 대조/정정
    # (관전 모드는 우리/상대 개념이 없어 side 매핑이 성립 안 함 → 스킵)
    if observer:
        print("[OCR 대조] 관전 모드 — 스킵 (휴리스틱 골 추론만 사용)")
    else:
        df = reconcile_ocr_goals(df, Path(csv_in).parent / "goals_ocr.json")

    events = df[df["action"] == "pass_or_shot"]
    print(f"[이벤트] pass_or_shot 총 {len(events)}개")
    print(events["outcome"].value_counts().to_string())

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"\n[저장] {csv_out}")


if __name__ == "__main__":
    import sys
    observer = "--observer" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_in = args[0] if len(args) > 0 else DEFAULT_CSV_IN
    csv_out = args[1] if len(args) > 1 else DEFAULT_CSV_OUT
    main(csv_in, csv_out, observer=observer)

"""
Step 3d — 영상 전체 → game_state.csv

설계 핵심:
  1. init_team_colors: 영상 여러 시점에서 K-Means 시도 후
     두 색이 가장 멀리 떨어진 결과 채택 → 팀 색 고정
  2. label_dots: 매 프레임 새로 K-Means 안 돌리고 고정된 두 색과
     거리 비교 → 라벨 일관성 유지 (T0 = 항상 같은 팀)
  3. CSV 한 줄에 공, 조작선수, 양 팀 dot 좌표 다 저장 (JSON 문자열)

실행:
  python -m src.extract_game_state data/videos/match1.mp4
  python -m src.extract_game_state data/videos/match1.mp4 3   # 3프레임마다 (속도 ↑)
"""

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.minimap import load_roi, crop_minimap
from src.detect_ball import detect_ball
from src.detect_player import detect_red_dot
from src.detect_teams import (get_dot_pixels, extract_dot_colors,
                                estimate_team_colors, is_minimap_valid,
                                has_field_rect, load_manual_team_colors,
                                build_team_model, classify_dots_v2)


def init_team_model(video_path: str, n_attempts: int = 12):
    """팀 분리 모델 결정 (v2 — 2단계 분리).

    여러 시점 미니맵을 모아 build_team_model 로 합의 모델 도출.
    단일 max-dist 프레임 락인(구버전) 대신 분포 합의를 쓴다.
    """
    roi = load_roi(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []
    for i in range(n_attempts):
        idx = int(n_frames * (i + 1) / (n_attempts + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        mini = crop_minimap(frame, roi)
        if is_minimap_valid(mini):
            samples.append(mini)
    cap.release()

    model = build_team_model(samples)
    if model is None:
        raise RuntimeError("팀 모델 추정 실패. 미니맵 ROI 또는 영상 확인 필요.")
    return model


def init_team_colors(video_path: str, n_attempts: int = 10):
    """(구버전 호환용) 단일 max-dist 프레임 기반 K-Means. 신규 경로는 init_team_model 사용."""
    manual = load_manual_team_colors(video_path)
    if manual is not None:
        print(f"[수동 지정 색 사용] T0 BGR={manual[0]}, T1 BGR={manual[1]}")
        return manual

    roi = load_roi(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    best = None
    best_dist = 0
    for i in range(n_attempts):
        idx = int(n_frames * (i + 1) / (n_attempts + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        mini = crop_minimap(frame, roi)
        c1, c2 = estimate_team_colors(mini)
        if c1 is None or c2 is None:
            continue
        d = sum((a - b) ** 2 for a, b in zip(c1, c2))
        if d > best_dist:
            best_dist = d
            best = (c1, c2)
    cap.release()

    if best is None:
        raise RuntimeError("팀 색 추정 실패. 미니맵 ROI 또는 영상 확인 필요.")
    return best


def label_dots(colors_np, team_colors):
    """각 dot 색을 두 팀 색과 비교 → 가까운 쪽 라벨 (0 또는 1)."""
    c0 = np.array(team_colors[0])
    c1 = np.array(team_colors[1])
    d0 = np.sum((colors_np - c0) ** 2, axis=1)
    d1 = np.sum((colors_np - c1) ** 2, axis=1)
    return (d1 < d0).astype(int)


def is_strict_minimap_valid(mini_bgr, grass_threshold: float = 0.5,
                              min_dot_count: int = 10) -> bool:
    """엄격한 valid: 잔디 50%+ 와 작은 dot 10개+. 짤린 미니맵/하이라이트 거름."""
    if not is_minimap_valid(mini_bgr, grass_threshold=grass_threshold):
        return False
    _, keep = get_dot_pixels(mini_bgr)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_dots = [c for c in contours if 3 <= cv2.contourArea(c) <= 50]
    return len(valid_dots) >= min_dot_count


def extract_frame_state(mini_bgr, team_model, strict: bool = False):
    """한 프레임의 게임 상태 dict. 미니맵 invalid면 모든 값 None/빈 리스트.

    team_model: build_team_model() 결과 dict (v2 2단계 분리).
    strict=True: 엄격 valid (관전 영상의 짤린 미니맵 거름).
    """
    valid_fn = is_strict_minimap_valid if strict else is_minimap_valid
    if not valid_fn(mini_bgr):
        return {"ball": None, "player": None, "t0_dots": [], "t1_dots": [], "valid": False}

    ball = detect_ball(mini_bgr)
    player = detect_red_dot(mini_bgr)
    t0, t1 = classify_dots_v2(mini_bgr, team_model)

    # 빨강(조작선수) 회수: 빨강 dot은 어느 팀에서 1명이 빨강으로 칠해져
    # 분류에서 빠진 것 → 그 팀이 1명 적다. 색을 모르니 '더 적은 팀'에 되돌린다.
    # (자기교정, 색 가정 불필요. 단 이미 잡힌 dot과 겹치면 중복 방지 위해 거리 체크)
    if player is not None:
        def _near(pt, lst, r=4):
            return any(abs(pt[0]-x) <= r and abs(pt[1]-y) <= r for x, y in lst)
        if not _near(player, t0) and not _near(player, t1):
            if len(t0) <= len(t1):
                t0.append((int(player[0]), int(player[1])))
            else:
                t1.append((int(player[0]), int(player[1])))

    return {"ball": ball, "player": player, "t0_dots": t0, "t1_dots": t1, "valid": True}


def process_video(video_path: str,
                  out_csv: str = "data/output/game_state.csv",
                  sample_every: int = 3,
                  strict: bool = False,
                  use_yolo: bool = False):
    # Fix4 — ROI 미보정 자동 감지 + 자동 해결:
    # 영상 전용 ROI 가 없으면, 기존에 등록된 ROI 후보들을 샘플 프레임으로
    # 검증해서 맞는 레이아웃을 자동 등록 (같은 해상도 재녹화 대응).
    # 어떤 후보도 안 맞으면 그때 중단.
    from src.minimap import list_rois, save_roi
    import json as _json
    stem = Path(video_path).stem
    if stem not in list_rois():
        print(f"[ROI 자동 탐색] '{stem}' ROI 없음 → 알려진 레이아웃 후보 검증")
        rois_all = _json.loads(
            (Path(__file__).resolve().parent.parent / "data" / "minimap_roi.json")
            .read_text())["rois"]
        # 중복 제거한 후보 목록
        cands = []
        for v in rois_all.values():
            if tuple(v) not in [tuple(c) for c in cands]:
                cands.append(v)
        cap0 = cv2.VideoCapture(video_path)
        nf = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
        W0, H0 = int(cap0.get(3)), int(cap0.get(4))
        best, best_ok = None, 0
        for cand in cands:
            x, y, w, h = cand
            if x + w > W0 or y + h > H0:
                continue
            ok_c = 0
            for k in range(10):
                cap0.set(cv2.CAP_PROP_POS_FRAMES, int(nf * (k + 1) / 11))
                ok, fr = cap0.read()
                if not ok:
                    continue
                mm = fr[y:y + h, x:x + w]
                if mm.size and is_minimap_valid(mm) and has_field_rect(mm):
                    ok_c += 1
            if ok_c > best_ok:
                best, best_ok = cand, ok_c
        cap0.release()
        if best is not None and best_ok >= 3:
            save_roi(tuple(best), video_path=video_path)
            print(f"[ROI 자동 등록] {stem} ← {best} (샘플 {best_ok}/10 유효)")
        else:
            raise RuntimeError(
                f"ROI 미보정: '{stem}' 영상에 맞는 알려진 레이아웃이 없습니다.\n"
                f"새 레이아웃이면 수동 보정 필요: python -m src.minimap calibrate {video_path}"
            )

    yolo = None
    team_model = None
    if use_yolo:
        from src.detect_yolo import YoloDotDetector, _MODEL_PATH
        yolo = YoloDotDetector()
        print(f"[초기화] YOLO dot 검출 모드 ({_MODEL_PATH})")
        yolo.init_ref_colors(video_path, load_roi(video_path=video_path))
    else:
        print("[초기화] 팀 분리 모델 추정 (여러 시점 합의)...")
        team_model = init_team_model(video_path)
        print(f"[초기화] team_model: {team_model}")

    roi = load_roi(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # ROI 유효성 사전 검사 — 영상이 교체됐는데 옛 ROI 가 남아 있으면
    # 조용히 전 프레임 invalid 쓰레기가 생성됨 (pro01/match03 사고 재발 방지)
    ok_cnt = 0
    for k in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * (k + 1) / 11))
        ok, fr = cap.read()
        if not ok:
            continue
        mm = crop_minimap(fr, roi)
        if mm.size and is_minimap_valid(mm) and has_field_rect(mm):
            ok_cnt += 1
    if ok_cnt == 0:
        cap.release()
        raise RuntimeError(
            f"ROI 불일치: 샘플 10프레임 전부 미니맵 검증 실패.\n"
            f"영상이 교체/재녹화됐다면 ROI 재보정 필요: "
            f"python -m src.minimap calibrate {video_path}\n"
            f"현재 ROI: {roi}")
    print(f"[ROI 검사] 샘플 10프레임 중 {ok_cnt}개 유효 → 통과")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[처리] {n_frames}프레임, {fps:.1f}fps, sample_every={sample_every}")

    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            mini = crop_minimap(frame, roi)
            if yolo is not None:
                # YOLO 경로 — 미니맵 유효할 때만 (리플레이/컷신/전환화면 제외)
                if is_minimap_valid(mini) and has_field_rect(mini):
                    state = yolo.detect(mini)
                    # dot 모델은 조작선수(빨강)를 별도 클래스로 안 잡음
                    # → 색 검출로 보강 (기본 모드 소유권 판정에 필요)
                    if state["player"] is None:
                        state["player"] = detect_red_dot(mini)
                else:
                    state = {"ball": None, "player": None, "t0_dots": [], "t1_dots": [], "valid": False}
            else:
                state = extract_frame_state(mini, team_model, strict=strict)
            records.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "minimap_valid": state.get("valid", False),
                "ball_x": state["ball"][0] if state["ball"] else None,
                "ball_y": state["ball"][1] if state["ball"] else None,
                "player_x": state["player"][0] if state["player"] else None,
                "player_y": state["player"][1] if state["player"] else None,
                "t0_count": len(state["t0_dots"]),
                "t1_count": len(state["t1_dots"]),
                "t0_dots": json.dumps(state["t0_dots"]),
                "t1_dots": json.dumps(state["t1_dots"]),
            })

        frame_idx += 1
        if frame_idx % 1000 == 0:
            pct = frame_idx * 100 // n_frames

    cap.release()

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 메타데이터 (팀 모델, fps 등) 별도 JSON
    meta_path = Path(out_csv).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "team_model": team_model,
        "detector": "yolo" if use_yolo else "colormask",
        "sample_every": sample_every,
        "fps": fps,
        "total_frames": n_frames,
        "roi": list(roi),   # grid.py 가 미니맵 크기를 정확히 알 수 있게
    }, indent=2, ensure_ascii=False))

    # 요약
    ball_ok = df["ball_x"].notna().sum()
    player_ok = df["player_x"].notna().sum()
    avg_t0 = df["t0_count"].mean()
    avg_t1 = df["t1_count"].mean()
    print(f"\n[완료] {len(df)}레코드")
    print(f"  공 검출: {ball_ok}/{len(df)} ({ball_ok*100//len(df)}%)")
    print(f"  조작선수 검출: {player_ok}/{len(df)} ({player_ok*100//len(df)}%)")
    print(f"  평균 dot 수: T0={avg_t0:.1f}, T1={avg_t1:.1f}")
    print(f"[저장] {out_csv}")
    print(f"[메타] {meta_path}")


if __name__ == "__main__":
    import sys
    strict = "--strict-valid" in sys.argv
    use_yolo = "--yolo" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video = args[0] if len(args) > 0 else "data/videos/match1.mp4"
    sample_every = int(args[1]) if len(args) > 1 else 3
    process_video(video, sample_every=sample_every, strict=strict, use_yolo=use_yolo)

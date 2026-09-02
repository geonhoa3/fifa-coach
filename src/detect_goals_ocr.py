"""
스코어보드 OCR 골 검출 — goals_ocr.json 생성

원리:
  FC 온라인 HUD 좌상단 스코어 배너 (팀명 [n] [m] 팀명 시계) 는
  경기 내내 같은 위치에 뜬다. 숫자 두 칸만 주기적으로 읽어서
  점수 변화 시점 = 골 시점을 확정한다.

  미니맵 휴리스틱(infer_goals)보다 확실한 ground truth:
  - 선방+리플레이를 골로 오탐하는 문제 없음
  - 리플레이 중엔 배너가 숨어 빈 값 → 자연스럽게 무시됨

흐름:
  1. calibrate_score_cells: 초반 프레임 몇 개에서 EasyOCR readtext 로
     숫자 토큰 페어(점수 칸) 위치 합의 → 셀 좌표 고정
     (EasyOCR 검출 단계는 외딴 '1' 을 자주 놓침 → 이후엔 검출 없이
      고정 셀을 recognize() 로만 읽는다)
  2. scan_scores: every_sec 간격으로 두 셀 recognize → 점수 시계열
  3. extract_goals: 연속 2표본 이상 유지되는 +1 변화만 골로 인정
     (골 시각은 [직전 구점수 관측, 첫 신점수 관측] 윈도우로 저장)

실행:
  python -m src.detect_goals_ocr data/videos/match08.mp4
  python -m src.detect_goals_ocr data/videos/match08.mp4 --every 3
"""

import cv2
import json
import numpy as np
from pathlib import Path


DEFAULT_OUT = "data/output/goals_ocr.json"


def _get_reader():
    import easyocr
    return easyocr.Reader(["ko", "en"], gpu=False, verbose=False)


def calibrate_score_cells(video_path: str, reader,
                           n_frames: int = 5,
                           strip_h_frac: float = 0.12,
                           strip_w_frac: float = 0.5,
                           pad_x: int = 8, pad_y: int = 6):
    """스코어 숫자 두 칸의 프레임 좌표 [x0,x1,y0,y1] 페어 반환. 실패 시 None.

    숫자 토큰(1~2자리, ':' 없는 것) 중 같은 행에서 가장 가까운 페어를
    여러 프레임에서 수집 → 위치 합의(±10px)가 과반이면 채택.
    """
    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(3)), int(cap.get(4))
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 1차: 초반 10~70초. 실패 시 2차: 영상 전체에 분산
    # (인트로 컷신으로 시작하는 녹화는 초반에 배너가 없음 — match17 사례)
    duration = nf / fps
    time_sets = [
        [10 + k * 15 for k in range(n_frames)],
        [duration * f for f in (0.25, 0.4, 0.55, 0.7, 0.85)],
    ]
    for attempt, times in enumerate(time_sets):
        cells = _calibrate_at(cap, times, W, fps, nf, reader, pad_x, pad_y)
        if cells is not None:
            if attempt == 1:
                print("[캘리브레이션] 초반 실패 → 전체 구간 재시도로 성공")
            cap.release()
            return cells
    cap.release()
    return None


def _calibrate_at(cap, times, W, fps, nf, reader, pad_x, pad_y,
                   strip_h_frac: float = 0.12, strip_w_frac: float = 0.5):
    H = int(cap.get(4))
    pairs = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(t * fps, nf - 1)))
        ok, fr = cap.read()
        if not ok:
            continue
        strip = fr[0:int(H * strip_h_frac), 0:int(W * strip_w_frac)]
        toks = []
        for box, txt, conf in reader.readtext(strip):
            if txt.isdigit() and 1 <= len(txt) <= 2 and conf > 0.5:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                toks.append((min(xs), max(xs), min(ys), max(ys)))
        # 같은 행 + 가장 가까운 페어
        best = None
        for a in toks:
            for b in toks:
                if b[0] <= a[0]:
                    continue
                cy_a, cy_b = (a[2] + a[3]) / 2, (b[2] + b[3]) / 2
                h_a = a[3] - a[2]
                gap = b[0] - a[1]
                if abs(cy_a - cy_b) < h_a and 5 < gap < 0.06 * W:
                    if best is None or gap < best[2]:
                        best = (a, b, gap)
        if best:
            pairs.append((best[0], best[1]))

    if not pairs:
        return None
    # 위치 합의: 왼쪽 셀 x0 기준 ±10px 클러스터 최빈
    ref = pairs[0]
    agree = [p for p in pairs
             if abs(p[0][0] - ref[0][0]) <= 10 and abs(p[1][0] - ref[1][0]) <= 10]
    if len(agree) < max(2, (len(pairs) + 1) // 2):
        return None
    cells = []
    for side in (0, 1):
        # 안쪽(두 칸 사이) 패딩은 2px 로 제한 — 넉넉히 잡으면 칸 사이
        # 구분선이 '1' 로 판독돼 점수가 2자리로 읽힘 (match23 13:2 오독 사고)
        inner = 2
        pl = pad_x if side == 0 else inner
        pr = inner if side == 0 else pad_x
        x0 = int(np.median([p[side][0] for p in agree])) - pl
        x1 = int(np.median([p[side][1] for p in agree])) + pr
        y0 = int(np.median([p[side][2] for p in agree])) - pad_y
        y1 = int(np.median([p[side][3] for p in agree])) + pad_y
        cells.append([max(0, x0), x1, max(0, y0), y1])
    return cells


def _banner_visible(grey, cells, dark_thr: float = 90, bright_thr: float = 140):
    """스코어 배너가 실제로 떠 있는지 검증.

    진짜 배너: 점수 칸 = 밝은 박스(~215+), 양옆 팀명 바 = 어두움(~30).
    이 검증 없이는 메뉴/구단정보/세리머니 오버레이 화면의 숫자가
    셀 위치에 지속 판독되어 유령 골이 생김 (match08 종료 메뉴 2:2 사례).
    실측: 리플레이 숨김(전부~60), 전술 메뉴(~22), 구단정보(우바 212),
    세리머니 오버레이(전부 152) 모두 걸러짐.
    """
    (x0l, x1l, y0l, y1l), (x0r, x1r, y0r, y1r) = cells
    h = max(1, y1l - y0l)
    left_bar = grey[y0l:y1l, max(0, x0l - 3 * h):max(1, x0l - 8)]
    right_bar = grey[y0r:y1r, x1r + 8:x1r + 8 + 3 * h]
    cell_l = grey[y0l:y1l, x0l:x1l]
    cell_r = grey[y0r:y1r, x0r:x1r]
    if left_bar.size == 0 or right_bar.size == 0 \
            or cell_l.size == 0 or cell_r.size == 0:
        return False
    return (left_bar.mean() < dark_thr and right_bar.mean() < dark_thr
            and cell_l.mean() > bright_thr and cell_r.mean() > bright_thr)


def read_team_names(video_path: str, cells, reader):
    """스코어 배너의 좌/우 팀명 OCR. {'left':.., 'right':..} 또는 None.

    reconcile 의 side→우리 매핑을 소유권 다수결(오판 가능) 대신
    닉네임으로 앵커하기 위함 (match13에서 다수결 1:4 반전 사고).
    배너가 보이는 시점을 골라 읽음 — 고정 30초는 인트로 컷신이나
    리플레이에 걸려 오독됨 (match16 '갈채(MF)' 사례).
    """
    cap = cv2.VideoCapture(video_path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = nf / fps
    fr = None
    for t_sec in [30, duration * 0.3, duration * 0.5, duration * 0.7]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
        ok, cand = cap.read()
        if not ok:
            continue
        grey = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        if _banner_visible(grey, cells):
            fr = cand
            break
    cap.release()
    if fr is None:
        return None
    H, W = fr.shape[:2]
    y0 = max(0, cells[0][2] - 15)
    y1 = min(H, cells[0][3] + 15)
    strip = fr[y0:y1, 0:int(W * 0.5)]
    toks = []
    for box, txt, conf in reader.readtext(strip):
        if conf < 0.3 or ":" in txt or txt.strip().isdigit():
            continue
        xs = [p[0] for p in box]
        toks.append((min(xs), max(xs), txt.strip()))
    left = [t for t in toks if t[1] <= cells[0][0]]
    right = [t for t in toks if t[0] >= cells[1][1]]
    return {
        "left": max(left, key=lambda t: t[1])[2] if left else None,
        "right": min(right, key=lambda t: t[0])[2] if right else None,
    }


def _read_cell(grey, cell, reader, min_conf: float = 0.4):
    """고정 셀 하나 recognize. 숫자 문자열 또는 None(배너 숨김/저신뢰)."""
    x0, x1, y0, y1 = cell
    img = grey[y0:y1, x0:x1]
    if img.size == 0:
        return None
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    r = reader.recognize(img,
                         horizontal_list=[[0, img.shape[1], 0, img.shape[0]]],
                         free_list=[], allowlist="0123456789", detail=1)
    if not r:
        return None
    txt, conf = r[0][1], r[0][2]
    if txt.isdigit() and conf >= min_conf:
        return txt
    return None


def scan_scores(video_path: str, cells, reader, every_sec: float = 2.0):
    """every_sec 간격으로 두 셀을 읽어 점수 시계열 반환."""
    cap = cv2.VideoCapture(video_path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = nf / fps

    readings = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if _banner_visible(grey, cells):
            l = _read_cell(grey, cells[0], reader)
            r = _read_cell(grey, cells[1], reader)
        else:
            l = r = None   # 배너 숨김/메뉴/오버레이 — 판독 안 함
        readings.append({"t": round(t, 1), "left": l, "right": r})
        t += every_sec
    cap.release()
    return readings


def extract_goals(readings, hold_n: int = 5, hold_frac: float = 0.6):
    """점수 시계열 → 골 목록 (오프라인 다수결 확인).

    좌우 독립 처리. 각 사이드의 점수는 단조 비감소여야 함.
    증가 후보(v > cur)는 그 시점부터 앞으로 hold_n 개 유효 표본 중
    hold_frac 이상이 v 와 일치할 때만 수락.

    이렇게 하면 골 리플레이 오버레이의 짧은 오독(2표본 지속)이
    유령 골로 수락되고 이후 정상 판독이 전부 '감소'로 거부되는 사고 방지
    (match05: 272s 유령 골 → 282~416s 판독 140초 거부 사례).

    골 시각은 (직전 구점수 관측 t_before, 첫 신점수 관측 t_after) 윈도우 —
    실제 골은 이 사이 (보통 t_after 직전, 리플레이 전에 점수 갱신됨).
    """
    goals = []
    init = {"left": 0, "right": 0}
    for side in ("left", "right"):
        seq = [(rd["t"], int(rd[side])) for rd in readings
               if rd[side] is not None and rd["left"] is not None
               and rd["right"] is not None]
        if len(seq) < 3:
            continue
        # 초기값: 앞쪽 5표본 다수결 (보통 0, 중간부터 녹화면 그 시점 점수)
        head = [v for _, v in seq[:5]]
        cur = max(set(head), key=head.count)
        init[side] = cur
        last_t = seq[0][0]
        k = 0
        while k < len(seq):
            t, v = seq[k]
            if v == cur:
                last_t = t
                k += 1
                continue
            if v < cur:
                k += 1  # 감소 = 오독, 무시
                continue
            # 증가 후보 — 앞으로 hold_n 유효 표본 다수결로 확인
            future = [x for _, x in seq[k:k + hold_n]]
            need = max(2, int(np.ceil(hold_frac * len(future))))
            if sum(1 for x in future if x == v) >= need:
                delta = v - cur
                if delta > 2:
                    print(f"[경고] {side} {cur}→{v} ({t:.0f}s) — "
                          f"3+ 점프는 오독 가능성, 2골만 인정")
                    delta = 2
                for _ in range(delta):
                    goals.append({"t_before": last_t, "t_after": t,
                                  "side": side, "new_val": v})
                if delta > 1:
                    print(f"[경고] {last_t:.0f}~{t:.0f}s 사이 {side} "
                          f"{delta}골 — 배너 공백 중 연속 골?")
                cur = v
                last_t = t
            else:
                print(f"[경고] {side} {cur}→{v} ({t:.0f}s) 후보가 이후 표본에서 "
                      f"유지 안 됨 — 오독으로 무시")
            k += 1

    # 시간순 정렬 + 누적 스코어 문자열 부여
    goals.sort(key=lambda g: g["t_after"])
    score = dict(init)
    for g in goals:
        score[g["side"]] += 1
        g["score"] = f"{score['left']}:{score['right']}"
        g.pop("new_val", None)
    return goals


def main(video_path: str, out_json: str = DEFAULT_OUT, every_sec: float = 2.0):
    print(f"[OCR 골 검출] {video_path}")
    try:
        reader = _get_reader()
    except ImportError:
        print("[스킵] easyocr 미설치 — OCR 골 검출 없이 진행 "
              "(pip install easyocr)")
        return

    cells = calibrate_score_cells(video_path, reader)
    if cells is None:
        print("[스킵] 스코어 배너 캘리브레이션 실패 — "
              "초반 프레임에 점수 표시가 없거나 레이아웃이 다름")
        return
    print(f"[캘리브레이션] 점수 셀: left={cells[0]}, right={cells[1]}")

    team_names = read_team_names(video_path, cells, reader)
    if team_names:
        print(f"[팀명] left={team_names['left']}, right={team_names['right']}")

    readings = scan_scores(video_path, cells, reader, every_sec=every_sec)
    n_valid = sum(1 for r in readings if r["left"] is not None)
    print(f"[스캔] {len(readings)}표본 중 판독 {n_valid}개 "
          f"(간격 {every_sec}s)")

    goals = extract_goals(readings)
    final = goals[-1]["score"] if goals else "0:0"
    print(f"[골] {len(goals)}개, 최종 스코어 {final}")
    for g in goals:
        t = g["t_after"]
        print(f"  {int(t // 60)}:{int(t % 60):02d}  {g['side']:5s} 득점 "
              f"→ {g['score']}  (윈도우 {g['t_before']:.0f}~{g['t_after']:.0f}s)")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps({
        "video": video_path,
        "cells": cells,
        "team_names": team_names,
        "every_sec": every_sec,
        "final_score": final,
        "goals": goals,
        "readings": readings,   # 원본 판독 — 알고리즘 개선 시 재스캔 없이 재추출
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[저장] {out_json}")


def reextract(json_path: str):
    """저장된 readings 로 골 추출만 다시 수행 (영상 재스캔 없음)."""
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    readings = data.get("readings")
    if not readings:
        print(f"[스킵] {json_path} 에 readings 없음 — 전체 재스캔 필요")
        return
    goals = extract_goals(readings)
    data["goals"] = goals
    data["final_score"] = goals[-1]["score"] if goals else "0:0"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[재추출] {json_path}: 골 {len(goals)}개, "
          f"최종 {data['final_score']}")
    for g in goals:
        t = g["t_after"]
        print(f"  {int(t // 60)}:{int(t % 60):02d}  {g['side']:5s} → {g['score']}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--from-json" in args:
        i = args.index("--from-json")
        reextract(args[i + 1])
        sys.exit(0)
    every = 2.0
    if "--every" in args:
        i = args.index("--every")
        every = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    video = args[0] if args else "data/videos/match08.mp4"
    out = args[1] if len(args) > 1 else DEFAULT_OUT
    main(video, out, every_sec=every)

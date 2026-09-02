"""
Step 3b — 두 팀 dot 색 자동 추정 (K-Means K=2)

알고리즘:
  1. 미니맵에서 잔디 / 노란(공) / 빨강(조작 선수) 픽셀 제외
  2. 남은 픽셀 = dot 후보
  3. K-Means K=2 로 두 클러스터 중심 색 추정 → 두 팀 색

실행:
  python -m src.detect_teams data/videos/match1.mp4 9803
"""

import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from src.minimap import load_roi, crop_minimap, grab_frame
from src.detect_ball import YELLOW_LOW, YELLOW_HIGH
from src.detect_player import RED_LOW1, RED_HIGH1, RED_LOW2, RED_HIGH2


TEAM_COLORS_DIR = Path("data/team_colors")


def load_manual_team_colors(video_path: str):
    """영상별 수동 지정 팀 색 로드. 파일 없으면 None.

    파일 구조: {"t0_bgr": [B, G, R], "t1_bgr": [B, G, R]}
    """
    name = Path(video_path).stem
    p = TEAM_COLORS_DIR / f"{name}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return tuple(data["t0_bgr"]), tuple(data["t1_bgr"])


def save_manual_team_colors(video_path: str, t0_bgr: tuple, t1_bgr: tuple):
    name = Path(video_path).stem
    p = TEAM_COLORS_DIR / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "t0_bgr": list(t0_bgr), "t1_bgr": list(t1_bgr),
    }, indent=2))
    print(f"[저장] {p}: T0={t0_bgr}, T1={t1_bgr}")


def pick_team_colors(video_path: str, frame_idx: int = None):
    """미니맵 한 프레임 띄움 → 사용자가 T0 dot, T1 dot 순서로 클릭 → BGR 저장.

    조작:
      - 마우스 클릭으로 dot 선택 (3x3 평균 색 추출)
      - 1번째 클릭 = T0, 2번째 클릭 = T1
      - r 키: 리셋 (다시 클릭)
      - ENTER 또는 s 키: 저장 + 종료
      - q 키: 저장 안 하고 종료
    """
    if frame_idx is None:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = n_frames // 3   # 1/3 지점 (게임 진행 중일 확률 높음)

    frame = grab_frame(video_path, frame_idx=frame_idx)
    roi = load_roi(video_path=video_path)
    mini = crop_minimap(frame, roi).copy()
    base = mini.copy()
    h, w = mini.shape[:2]

    # 4배 확대로 작은 dot 클릭 쉽게
    SCALE = 4
    big = cv2.resize(base, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    # dot 후보 자동 검출 (사용자에게 "여기 dot 있어요" 표시용)
    _, keep = get_dot_pixels(base)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 1 <= area <= 50:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dot_candidates.append((cx, cy))
    print(f"[자동 검출] dot 후보 {len(dot_candidates)}개 — 녹색 원 안에 클릭")

    state = {"clicks": []}  # [(bgr_tuple, (ox, oy)), ...]

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(state["clicks"]) >= 2:
            return
        # 원본 좌표로 변환
        ox, oy = x // SCALE, y // SCALE
        # 더 넓은 patch 7x7 (정확한 dot 가운데 못 잡아도 OK)
        y0, y1 = max(0, oy - 3), min(h, oy + 4)
        x0, x1 = max(0, ox - 3), min(w, ox + 4)
        patch = base[y0:y1, x0:x1].reshape(-1, 3)

        # 잔디 + 검정 자동 제외. 단, 밝은 흰색은 보호 (V >= 180 이면 잔디 아님).
        # 검정 outline (V < 60) 도 제외 — 흰 ring dot 의 검은 외곽이 평균 끌어내림 방지
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grass_mask = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 30) & (hsv[:, 2] < 180)
        black_mask = hsv[:, 2] < 60
        exclude_mask = grass_mask | black_mask
        non_grass = patch[~exclude_mask]

        if len(non_grass) >= 3:
            bgr = tuple(int(v) for v in non_grass.mean(axis=0))
            note = f"(잔디 {grass_mask.sum()}/{len(patch)}개 제외)"
        else:
            bgr = tuple(int(v) for v in patch.mean(axis=0))
            note = "(잔디 제외 안 됨 — patch에 dot 픽셀 부족)"

        state["clicks"].append((bgr, (ox, oy)))
        label = f"T{len(state['clicks']) - 1}"
        print(f"[클릭 {len(state['clicks'])}/2] {label} BGR={bgr} {note} (좌표 {ox},{oy})")

    win = "Pick team colors: 1st click=T0, 2nd click=T1 | r=reset, s/ENTER=save, q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(win, on_click)
    print(f"[안내] T0(첫 팀) dot 한 번, T1(둘째 팀) dot 한 번 클릭")
    print(f"       리셋: r,  저장: s 또는 ENTER,  취소: q")

    while True:
        viz = big.copy()
        # dot 후보 자동 마커 (녹색 작은 원) — 사용자가 어디 클릭할지 보임
        for cx, cy in dot_candidates:
            mx, my = cx * SCALE + SCALE // 2, cy * SCALE + SCALE // 2
            cv2.circle(viz, (mx, my), SCALE + 2, (0, 255, 0), 1)
        # 클릭 위치에 마커 + 색 패치
        for i, (bgr, (ox, oy)) in enumerate(state["clicks"]):
            # 클릭한 원본 좌표 → 4배 확대 좌표
            mx, my = ox * SCALE + SCALE // 2, oy * SCALE + SCALE // 2
            cv2.drawMarker(viz, (mx, my), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(viz, f"T{i}", (mx + 12, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # 색 패치 (상단)
            cv2.rectangle(viz, (10 + i * 220, 10),
                          (200 + i * 220, 50), bgr, -1)
            cv2.putText(viz, f"T{i}={bgr}", (15 + i * 220, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(win, viz)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            print("[취소] 저장 안 함")
            return
        if k == ord('r'):
            state["clicks"] = []
            print("[리셋]")
        if k in (ord('s'), 13):  # 13 = ENTER
            if len(state["clicks"]) < 2:
                print(f"[대기] dot 두 개 클릭 필요 (현재 {len(state['clicks'])}/2)")
                continue
            break

    cv2.destroyAllWindows()
    t0_bgr = state["clicks"][0][0]
    t1_bgr = state["clicks"][1][0]
    save_manual_team_colors(video_path, t0_bgr, t1_bgr)


# HSV 잔디(녹색) 범위
GRASS_LOW = np.array([35, 30, 30])
GRASS_HIGH = np.array([85, 255, 230])


def is_minimap_valid(mini_bgr, grass_threshold: float = 0.4) -> bool:
    """ROI 영역에 잔디(녹색) 비율이 threshold 이상이면 진짜 미니맵."""
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    grass_ratio = (grass > 0).sum() / grass.size
    return grass_ratio >= grass_threshold


def has_field_rect(mini_bgr, h_frac=0.25, v_frac=0.25) -> bool:
    """흰 필드 외곽 직사각형 검사 (행/열 투영 방식).

    상/하단 30% 밴드에 흰 픽셀 비율 h_frac+ 인 행이,
    좌/우 30% 밴드에 v_frac+ 인 열이 각각 존재해야 진짜 미니맵.
    스코어보드·스킬팝업·잔디 클로즈업·페이드 중 미니맵을 거름.
    (실측: 정상 미니맵 4밴드 모두 0.33+, 전환화면 0.00~0.04)
    """
    H, W = mini_bgr.shape[:2]
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    white = (cv2.inRange(hsv, (0, 0, 130), (180, 100, 255)) > 0)
    rows = white.sum(axis=1) / W
    cols = white.sum(axis=0) / H
    t = int(H * 0.3); l = int(W * 0.3)
    return (rows[:t].max() >= h_frac and rows[-t:].max() >= h_frac
            and cols[:l].max() >= v_frac and cols[-l:].max() >= v_frac)

# HSV 검은색 범위 (dot 외곽선, 텍스트 등)
BLACK_LOW = np.array([0, 0, 0])
BLACK_HIGH = np.array([180, 255, 60])

# HSV 흰색 범위 (흰 팀 dot. 낮은 채도 + 높은 명도)
# 흰 ring dot 의 anti-aliasing 까지 잡으려고 임계 완화
WHITE_LOW = np.array([0, 0, 130])
WHITE_HIGH = np.array([180, 80, 255])


def get_dot_pixels(mini_bgr):
    """잔디/노란/빨강/검정 제외한 픽셀들과 keep 마스크 반환.
    흰 dot 은 잔디 마스크에 잘못 잡힐 수 있어 명시적으로 강제 포함.
    """
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)

    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    red1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    red2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    black = cv2.inRange(hsv, BLACK_LOW, BLACK_HIGH)
    white = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    exclude = cv2.bitwise_or(grass, yellow)
    exclude = cv2.bitwise_or(exclude, red1)
    exclude = cv2.bitwise_or(exclude, red2)
    exclude = cv2.bitwise_or(exclude, black)
    # 흰색은 잔디로 잘못 잡혔어도 다시 keep 으로 강제 포함
    exclude = cv2.bitwise_and(exclude, cv2.bitwise_not(white))
    keep = cv2.bitwise_not(exclude)

    pixels = mini_bgr[keep > 0]
    return pixels, keep


def extract_dot_colors(mini_bgr, keep_mask, min_area: int = 1, max_area: int = 50):
    """keep 마스크에서 각 dot(연결 영역)의 색을 모은다.

    중심 patch 대신 contour 내부 keep_mask 픽셀들의 평균.
    이유: 흰 ring dot 처럼 가운데가 비어있으면 중심 patch 가 잔디색이 됨.
          contour outline 픽셀 자체로 평균내면 ring dot 도 정확.
    """
    contours, _ = cv2.findContours(keep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colors = []
    centers = []
    h, w = mini_bgr.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # contour 내부 + keep_mask 픽셀 = 진짜 dot 픽셀 (ring 의 outline)
        contour_mask = np.zeros_like(keep_mask)
        cv2.drawContours(contour_mask, [c], -1, 255, -1)
        pixel_mask = (contour_mask > 0) & (keep_mask > 0)
        dot_pixels = mini_bgr[pixel_mask]
        if len(dot_pixels) == 0:
            continue
        colors.append(dot_pixels.mean(axis=0))
        centers.append((cx, cy))
    return np.array(colors), centers


def estimate_team_colors(mini_bgr):
    _, keep = get_dot_pixels(mini_bgr)
    colors, centers = extract_dot_colors(mini_bgr, keep)
    if len(colors) < 4:
        return None, None
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(colors)
    cc = km.cluster_centers_.astype(int)
    return tuple(int(v) for v in cc[0]), tuple(int(v) for v in cc[1])


def detect_all_dots(mini_bgr):
    _, keep = get_dot_pixels(mini_bgr)
    colors, centers = extract_dot_colors(mini_bgr, keep)
    if len(colors) < 4:
        return [], (None, None)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(colors)
    cc = km.cluster_centers_.astype(int)
    team_colors = (tuple(int(v) for v in cc[0]), tuple(int(v) for v in cc[1]))
    dots = [(cx, cy, int(lbl)) for (cx, cy), lbl in zip(centers, labels)]
    return dots, team_colors


# ════════════════════════════════════════════════════════════════════
#  v2 — 2단계 팀 분리 (흰/회색 팀 분리 → 남은 채도 dot hue 클러스터)
#
#  배경: FIFA 미니맵은 거의 항상 "한 팀=흰/회색, 다른 팀=채도 높은 색".
#        흰색은 hue가 노이즈라 BGR/hue 단독 클러스터링이 붕괴함.
#        → 흰 팀을 S/V로 먼저 떼고, 남은 dot만 hue(색상환)로 가른다.
# ════════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════════
#  v2 — 팀 분리 (밝은/어두운/유색 3-모드)
#  - bright_vs_dark : 흰 팀 vs 검정 팀 (명도로 분리) — pro01류
#  - white_vs_color : 흰 팀 vs 유색 팀 (채도/명도) — match03류
#  - two_color      : 유색 vs 유색 (hue 2분할)
# ════════════════════════════════════════════════════════════════════

WHITE_S_MAX = 40      # 채도 < 이 값 + 밝으면 흰/회색 팀
WHITE_V_MIN = 150
COLOR_S_MIN = 40      # 채도 >= 이 값이면 '유색' dot
DARK_S_MAX = 70       # 어두운 유니폼: 채도 낮고
DARK_V_MAX = 110      #               명도도 낮음 (검정/진회색 팀)


def extract_dot_core_colors(mini_bgr, min_area: int = 2, max_area: int = 60):
    """각 dot의 '채도 높은 코어' 픽셀 평균 BGR + 중심.
    잔디/노랑(공)/진짜검정외곽선만 픽셀단위 차단 (어두운 유니폼은 보존)."""
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    blackish = (hsv[:, :, 2] < 22).astype(np.uint8) * 255  # 외곽선/그림자만
    block = cv2.bitwise_or(cv2.bitwise_or(grass, yellow), blackish)
    cand = cv2.bitwise_not(block)

    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colors, centers = [], []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        m = np.zeros(cand.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        sel = (m > 0) & (block == 0)
        if sel.sum() < 2:
            continue
        s_vals = hsv[:, :, 1][sel]
        thr = max(40, np.percentile(s_vals, 60))
        core = sel & (hsv[:, :, 1] >= thr)
        if core.sum() < 1:
            core = sel
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        colors.append(mini_bgr[core].mean(axis=0))
        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # 어두운 dot(검정/암녹 유니폼)은 위 cand에서 큰 덩어리로 뭉쳐 누락됨.
    # 별도 마스크(잔디 색조이지만 어두운 픽셀)로 따로 검출해 합친다.
    dark_centers = _detect_dark_dots(mini_bgr, hsv)
    # 밝은 dot과 너무 가까운 건 중복 → 제외
    for dc in dark_centers:
        if all(abs(dc[0] - cx) > 3 or abs(dc[1] - cy) > 3 for cx, cy in centers):
            colors.append(np.array([30.0, 30.0, 30.0]))  # 더미 검정색 (명도로 분류됨)
            centers.append(dc)

    return np.array(colors), centers


def _detect_dark_dots(mini_bgr, hsv, vmax: int = 95, amin: int = 4,
                      amax: int = 130, circ_min: float = 0.55):
    """어두운 dot(검정/암녹/암갈 유니폼) 중심 좌표 리스트.
    명도(V)가 낮은 픽셀만 (색조 H 무관 — 주황/갈색 어두운 팀도 포함).
    잔디는 별도 마스크로 명시 제외. 그림자 띠는 원형도로 제거.
    """
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    # H 조건 제거: 어두운 dot은 색조 무관, V 낮고 채도 있는 것. 잔디만 빼면 됨.
    dark = ((V < vmax) & (S > 35) & (grass == 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < amin or a > amax:
            continue
        per = cv2.arcLength(c, True)
        if per == 0 or 4 * np.pi * a / (per * per) < circ_min:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return out


def _bgr_to_hsv_rows(colors_bgr):
    arr = colors_bgr.reshape(-1, 1, 3).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)


def _hue_circle_feats(H, S):
    h = H * 2 * np.pi / 180.0
    s = np.clip(S / 255.0, 0, 1)
    return np.stack([np.cos(h) * s, np.sin(h) * s], axis=1)


def build_team_model(samples_bgr):
    """여러 미니맵 프레임으로 팀 분리 모델 결정 (영상당 1회). 실패 시 None."""
    all_H, all_S, all_V = [], [], []
    for mini in samples_bgr:
        colors, _ = extract_dot_core_colors(mini)
        if len(colors) == 0:
            continue
        hsv = _bgr_to_hsv_rows(colors)
        all_H.append(hsv[:, 0]); all_S.append(hsv[:, 1]); all_V.append(hsv[:, 2])
    if not all_H:
        return None
    H = np.concatenate(all_H); S = np.concatenate(all_S); V = np.concatenate(all_V)

    # 어두운/밝은은 명도(V)만으로 판정 (어두운 유니폼은 채도 높아도 dark)
    bright = V > WHITE_V_MIN
    dark = V < DARK_V_MAX
    colored = S >= COLOR_S_MIN
    n_bright, n_dark, n_color = int(bright.sum()), int(dark.sum()), int(colored.sum())
    total = max(len(V), 1)

    # 1순위: 밝은 vs 어두운 (명도 분리) — 흰 vs 검정.
    # 명도 양극화가 뚜렷하면(중간대 V가 거의 없고 양쪽이 충분) 최우선 채택.
    # 검정팀이 '어두운 유색'이라 color로도 잡히는 케이스를 여기서 먼저 가로챈다.
    # 절대 개수 기준: 한 팀(흰)이 압도적이면 다른 팀(검정) 비율이 희석되므로
    # 비율이 아닌 '검정 dot 절대수 충분 + 중간대 거의 없음'으로 판정.
    mid = int(((V >= DARK_V_MAX) & (V <= WHITE_V_MIN)).sum())
    if n_bright >= 6 and n_dark >= 6 and mid <= 0.15 * total:
        v_split = (float(np.median(V[bright])) + float(np.median(V[dark]))) / 2
        return {"mode": "bright_vs_dark", "v_split": v_split}

    if n_color < 4:
        return None

    # 2순위: 흰 vs 유색 — 단 양 팀이 실제 인원 수준(각 5개+)일 때만.
    # (한쪽이 5개 미만이면 잘못된 분기 → two_color로 내려보냄)
    if n_bright >= 0.25 * (n_bright + n_color) and n_bright >= 5 and n_color >= 5:
        return {"mode": "white_vs_color", "color_hue": float(np.median(H[colored]))}

    # 3순위: 유색 vs 유색 (hue 2분할)
    feats = _hue_circle_feats(H[colored], S[colored])
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(feats)
    hueA = float(np.median(H[colored][km.labels_ == 0]))
    hueB = float(np.median(H[colored][km.labels_ == 1]))
    return {"mode": "two_color", "hueA": hueA, "hueB": hueB}


def _hue_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def classify_dots_v2(mini_bgr, model):
    """team_model 기준 각 dot을 T0/T1 분류. 반환 (t0_list, t1_list)."""
    colors, centers = extract_dot_core_colors(mini_bgr)
    if len(colors) == 0 or model is None:
        return [], []
    hsv = _bgr_to_hsv_rows(colors)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    t0, t1 = [], []
    for i, (cx, cy) in enumerate(centers):
        if model["mode"] == "bright_vs_dark":
            (t0 if V[i] >= model["v_split"] else t1).append((cx, cy))
        elif model["mode"] == "white_vs_color":
            if S[i] < WHITE_S_MAX and V[i] > WHITE_V_MIN:
                t0.append((cx, cy))
            elif S[i] >= COLOR_S_MIN:
                t1.append((cx, cy))
        else:
            if S[i] < 30:
                continue
            if _hue_diff(H[i], model["hueA"]) <= _hue_diff(H[i], model["hueB"]):
                t0.append((cx, cy))
            else:
                t1.append((cx, cy))
    return t0, t1

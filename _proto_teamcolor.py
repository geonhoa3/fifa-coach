"""기존 vs 신규 팀색 분리 비교 프로토타입 (검증용, 검증 후 삭제)."""
import cv2, numpy as np
from src.minimap import load_roi, crop_minimap
from src.detect_teams import (GRASS_LOW, GRASS_HIGH, get_dot_pixels,
                              extract_dot_colors, estimate_team_colors)
from src.detect_ball import detect_ball, YELLOW_LOW, YELLOW_HIGH
from src.detect_player import RED_LOW1, RED_HIGH1, RED_LOW2, RED_HIGH2
from sklearn.cluster import KMeans

# ── 신규: dot 코어 색 추출 (외곽선/잔디 오염 제거) ──
def extract_dot_core_colors(mini_bgr, min_area=2, max_area=60):
    """각 dot에서 '가장 채도 높은 코어' 픽셀들의 평균 BGR + 중심 반환.
    잔디/검정 외곽선이 평균을 오염시키지 않도록 dot 내부 상위채도만 채취."""
    hsv = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LOW, GRASS_HIGH)
    yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    blackish = (hsv[:, :, 2] < 70).astype(np.uint8) * 255
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
        s = hsv[:, :, 1][sel]
        thr = max(40, np.percentile(s, 60))   # 채도 상위 픽셀만 = dot 코어
        core = sel & (hsv[:, :, 1] >= thr)
        if core.sum() < 1:
            core = sel
        px = mini_bgr[core]
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        colors.append(px.mean(axis=0))
        centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
    return np.array(colors), centers

def hue_feats(colors_bgr):
    """BGR -> 색상환 2D 좌표 (채도 가중). 밝기 무시."""
    hsv = cv2.cvtColor(colors_bgr.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1,3).astype(float)
    h = hsv[:,0]*2*np.pi/180.0
    s = np.clip(hsv[:,1]/255.0, 0, 1)
    return np.stack([np.cos(h)*s, np.sin(h)*s], axis=1), hsv

def hue_dist_deg(c1_bgr, c2_bgr):
    f,_ = hue_feats(np.array([c1_bgr, c2_bgr], float))
    a = np.arctan2(f[0,1], f[0,0]); b = np.arctan2(f[1,1], f[1,0])
    d = abs(a-b)*180/np.pi
    return min(d, 360-d)

def new_estimate(mini_bgr):
    colors, centers = extract_dot_core_colors(mini_bgr)
    if len(colors) < 6:
        return None, None, 0
    feats, hsv = hue_feats(colors)
    km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(feats)
    cc = []
    for k in range(2):
        grp = colors[km.labels_==k]
        cc.append(tuple(int(v) for v in np.median(grp, axis=0)))
    return cc[0], cc[1], hue_dist_deg(cc[0], cc[1])

def grab(v, idx):
    cap=cv2.VideoCapture(v); cap.set(1,idx); ok,f=cap.read(); cap.release()
    return f if ok else None

for v in ["data/videos/match01.mp4","data/videos/match03.mp4","data/videos/pro_07.mp4"]:
    print(f"\n=== {v} ===")
    roi = load_roi(video_path=v)
    cap=cv2.VideoCapture(v); n=int(cap.get(7)); cap.release()
    old_d, new_d = [], []
    for i in range(1,9):
        f=grab(v,int(n*i/10))
        if f is None: continue
        mini=crop_minimap(f,roi)
        oc1,oc2=estimate_team_colors(mini)
        if oc1 and oc2: old_d.append(round(hue_dist_deg(oc1,oc2),1))
        nc1,nc2,nd=new_estimate(mini)
        if nc1: new_d.append(round(nd,1))
    print(f"  OLD hue분리(도): {old_d}  median={np.median(old_d) if old_d else 'NA'}")
    print(f"  NEW hue분리(도): {new_d}  median={np.median(new_d) if new_d else 'NA'}")

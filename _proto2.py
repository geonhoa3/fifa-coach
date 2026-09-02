"""2단계 팀 분리 검증: ①흰팀 분리 ②남은 채도 dot hue 클러스터. 검증 후 삭제."""
import cv2, numpy as np
from src.minimap import load_roi, crop_minimap
from _proto_teamcolor import extract_dot_core_colors

WHITE_S_MAX, WHITE_V_MIN = 40, 150   # 흰/회색 팀 기준

def classify_frame(mini_bgr):
    """반환: dots=[(cx,cy,team0|1)], desc=설명. team0=흰팀, team1=색팀(or hueA/B)."""
    colors, centers = extract_dot_core_colors(mini_bgr)
    if len(colors) < 6:
        return None, "dot<6"
    hsv = cv2.cvtColor(colors.reshape(-1,1,3).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).reshape(-1,3).astype(int)
    H,S,V = hsv[:,0],hsv[:,1],hsv[:,2]
    white = (S < WHITE_S_MAX) & (V > WHITE_V_MIN)
    colored = ~white & (S >= WHITE_S_MAX)        # 무딘 중간(잔디오염)은 양쪽 다 아님→버림
    nw, nc = white.sum(), colored.sum()
    # case A: 흰 vs 색  (둘 다 충분)
    if nw >= 3 and nc >= 3:
        labels = np.where(white, 0, np.where(colored, 1, -1))
        desc = f"흰({nw}) vs 색({nc})  버림{(labels==-1).sum()}"
    else:
        # case B: 색-색. 채도 있는 것만 hue로 2분할
        idx = np.where(S >= 30)[0]
        if len(idx) < 6:
            return None, f"분리불가 nw={nw} nc={nc}"
        from sklearn.cluster import KMeans
        h = H[idx]*2*np.pi/180; s=np.clip(S[idx]/255,0,1)
        feat = np.stack([np.cos(h)*s, np.sin(h)*s],1)
        km = KMeans(2,n_init=5,random_state=42).fit(feat)
        labels = np.full(len(colors),-1)
        labels[idx] = km.labels_
        a=(labels==0).sum(); b=(labels==1).sum()
        desc = f"색A({a}) vs 색B({b})"
    dots=[(centers[i][0],centers[i][1],int(labels[i])) for i in range(len(colors)) if labels[i]>=0]
    return dots, desc

def grab(v, idx):
    cap=cv2.VideoCapture(v); cap.set(1,idx); ok,f=cap.read(); cap.release(); return f if ok else None

for v in ["data/videos/match01.mp4","data/videos/match03.mp4","data/videos/pro_07.mp4"]:
    print(f"\n=== {v} ===")
    roi=load_roi(video_path=v); cap=cv2.VideoCapture(v); n=int(cap.get(7)); cap.release()
    for i in range(1,9):
        f=grab(v,int(n*i/10))
        if f is None: continue
        mini=crop_minimap(f,roi)
        dots,desc=classify_frame(mini)
        if dots is None: print(f"  f{i}: {desc}"); continue
        t0=sum(1 for _,_,t in dots if t==0); t1=sum(1 for _,_,t in dots if t==1)
        bal = min(t0,t1)/max(t0,t1) if max(t0,t1) else 0
        print(f"  f{i}: T0={t0:2d} T1={t1:2d} 균형={bal:.2f}  [{desc}]")

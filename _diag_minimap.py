import cv2, numpy as np, json, sys
from src.minimap import load_roi, crop_minimap
from src.detect_teams import (get_dot_pixels, extract_dot_colors,
                              estimate_team_colors, is_minimap_valid, GRASS_LOW, GRASS_HIGH)

def grab(video, idx):
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read(); cap.release(); return f if ok else None

def diag(video, nframes_to_check=8):
    cap = cv2.VideoCapture(video); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W=int(cap.get(3)); H=int(cap.get(4)); cap.release()
    try: roi = load_roi(video_path=video)
    except Exception as e: print(f"  ROI load fail: {e}"); return
    print(f"  res={W}x{H} frames={n} roi={roi}")
    valid=0; dotcounts=[]; colorpairs=[]
    for i in range(1,nframes_to_check+1):
        idx=int(n*i/(nframes_to_check+1)); f=grab(video,idx)
        if f is None: continue
        mini=crop_minimap(f,roi)
        # grass ratio
        hsv=cv2.cvtColor(mini,cv2.COLOR_BGR2HSV)
        gr=(cv2.inRange(hsv,GRASS_LOW,GRASS_HIGH)>0).mean()
        v=is_minimap_valid(mini)
        valid+= v
        _,keep=get_dot_pixels(mini); colors,centers=extract_dot_colors(mini,keep)
        dotcounts.append(len(centers))
        c1,c2=estimate_team_colors(mini)
        if c1 and c2:
            dist=sum((a-b)**2 for a,b in zip(c1,c2))**0.5
            colorpairs.append((c1,c2,round(dist,1)))
    print(f"  valid={valid}/{nframes_to_check}, grass(last)={gr:.2f}, dotcounts={dotcounts}")
    for c1,c2,d in colorpairs[:4]:
        print(f"    KMeans colors BGR {c1} vs {c2}  dist={d}")

for v in ["data/videos/match01.mp4","data/videos/match03.mp4","data/videos/pro_07.mp4"]:
    print(f"=== {v} ===")
    try: diag(v)
    except Exception as e: print("  ERR",e)

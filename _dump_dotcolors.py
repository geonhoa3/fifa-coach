import cv2, numpy as np
from src.minimap import load_roi, crop_minimap
import importlib.util
spec=importlib.util.spec_from_file_location("p","_proto_teamcolor.py")
# reuse extractor
from _proto_teamcolor import extract_dot_core_colors

def grab(v, idx):
    cap=cv2.VideoCapture(v); cap.set(1,idx); ok,f=cap.read(); cap.release()
    return f if ok else None

for v in ["data/videos/match01.mp4","data/videos/match03.mp4","data/videos/pro_07.mp4"]:
    print(f"\n=== {v} ===")
    roi=load_roi(video_path=v); cap=cv2.VideoCapture(v); n=int(cap.get(7)); cap.release()
    f=grab(v,int(n*0.4)); mini=crop_minimap(f,roi)
    colors,centers=extract_dot_core_colors(mini)
    if len(colors)==0:
        print("  no dots"); continue
    hsv=cv2.cvtColor(colors.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_BGR2HSV).reshape(-1,3)
    order=np.argsort(hsv[:,0])
    print(f"  {len(colors)} dots  (H,S,V) sorted by H:")
    for i in order:
        print(f"    H={hsv[i,0]:3d} S={hsv[i,1]:3d} V={hsv[i,2]:3d}  BGR={tuple(int(x) for x in colors[i])}")

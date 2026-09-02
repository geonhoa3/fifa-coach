"""
풀셋 v2 데이터셋 생성 — 해상도 3종 영상에서 프레임 추출 + 자동라벨 초안.

- pro01(흰vs검정), match03(흰vs빨강), pro_07(tan vs흰) 각 30장
- 자동라벨: 현재 18장 YOLO 모델로 초안 생성 (흰팀·공 위주, 못잡은 건 Roboflow에서 수동 추가)
- 본경기 화면/리플레이는 HoughLines(필드선)로 거름

실행: python make_dataset_v2.py
결과: data/yolo_v2/images, data/yolo_v2/labels  → Roboflow 업로드

라벨 클래스 (기존과 동일):
  0=ball, 1=team1, 2=team1_active, 3=team2, 4=team2_active
"""
import cv2, numpy as np, os
from src.minimap import load_roi, crop_minimap
from src.detect_teams import is_minimap_valid
from src.detect_yolo import YoloDetector

OUT_IMG = "data/yolo_v2/images"
OUT_LBL = "data/yolo_v2/labels"
VIDEOS = ["pro01", "match03", "pro_07"]
PER_VIDEO = 30


def looks_minimap(mini):
    """필드 라인(흰 직선) 2개+ 있으면 진짜 미니맵 (본경기/리플레이 거름)."""
    g = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 50, 150)
    L = cv2.HoughLinesP(e, 1, np.pi/180, 40,
                        minLineLength=mini.shape[1]//4, maxLineGap=10)
    return L is not None and len(L) >= 2


from src.detect_teams import has_field_rect  # 직사각형 검사 (canonical 위치: detect_teams)


def box(cx, cy, w, h, r=4):
    return f"{cx/w:.5f} {cy/h:.5f} {2*r/w:.5f} {2*r/h:.5f}"


def main():
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)
    det = YoloDetector(conf=0.25)
    total = 0
    for vid in VIDEOS:
        v = f"data/videos/{vid}.mp4"
        roi = load_roi(video_path=v)
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        for i in range(1, 120):
            if saved >= PER_VIDEO:
                break
            fi = int(n * i / 121)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if not ok:
                continue
            mini = crop_minimap(f, roi)
            if mini.size == 0 or not is_minimap_valid(mini) or not looks_minimap(mini) \
                    or not has_field_rect(mini):
                continue
            h, w = mini.shape[:2]
            name = f"{vid}_f{fi:06d}"
            cv2.imwrite(f"{OUT_IMG}/{name}.png", mini)
            st = det.detect(mini)
            lines = []
            if st["ball"]:
                lines.append(f"0 {box(st['ball'][0], st['ball'][1], w, h, 5)}")
            for cx, cy in st["t0_dots"]:
                lines.append(f"1 {box(cx, cy, w, h, 4)}")   # team1
            for cx, cy in st["t1_dots"]:
                lines.append(f"3 {box(cx, cy, w, h, 4)}")   # team2
            open(f"{OUT_LBL}/{name}.txt", "w").write("\n".join(lines) + "\n")
            saved += 1
        cap.release()
        print(f"  {vid}: {saved}장")
        total += saved
    # labelmap
    open(f"{OUT_LBL}/_classes.txt", "w").write(
        "ball\nteam1\nteam1_active\nteam2\nteam2_active\n")
    print(f"\n[완료] 총 {total}장 → data/yolo_v2/")
    print("  Roboflow에 images+labels 업로드 → 검정/tan팀 수동 보강 → 학습")


if __name__ == "__main__":
    main()

"""
FIFA 미니맵 dot 탐지 — YOLOv8 학습 (파일럿: pro01 18장)

사용법:
  1. Roboflow에서 받은 zip을 C:\\Project\\05_FIFA_COACH\\dataset\\ 에 풀기
     (안에 data.yaml 이 있어야 함)
  2. python train_yolo.py

설계 의도:
  - yolov8n (nano): dot처럼 단순·작은 객체엔 충분하고 빠름
  - imgsz=320: 미니맵이 308×155로 작으니 과하게 키울 필요 없음
  - epochs=150: 18장 소량이라 충분히 돌려 수렴 확인
  - 파일럿 목적 = "검정 dot이 색마스킹(1/11)보다 잘 잡히나" 검증
"""
from pathlib import Path
from ultralytics import YOLO
import torch

# ── 경로: zip 푼 위치의 data.yaml (실제 위치에 맞게 수정) ──
DATA_YAML = r"C:\Project\05_FIFA_COACH\dataset\data.yaml"


def main():
    dev = "0" if torch.cuda.is_available() else "cpu"
    print(f"[학습 장치] {'GPU ' + torch.cuda.get_device_name(0) if dev=='0' else 'CPU'}")

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"data.yaml 없음: {DATA_YAML}\n"
            f"Roboflow zip을 C:\\Project\\05_FIFA_COACH\\dataset\\ 에 풀었는지 확인."
        )

    model = YOLO("yolov8n.pt")   # nano 사전학습 가중치에서 시작 (transfer learning)

    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=320,          # 미니맵 크기에 맞춤 (작은 dot 보존)
        batch=8,
        device=dev,
        patience=50,        # 50epoch 개선 없으면 조기종료
        project="runs_fifa",
        name="pilot_pro01",
        # 소량 데이터 — 증강은 가볍게 (dot 위치/색 왜곡 최소)
        degrees=0, translate=0.05, scale=0.2, fliplr=0.5, mosaic=0.0,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,   # 색조는 거의 안 건드림(팀색 구분 보존)
    )
    print("\n[완료] 결과: runs_fifa/pilot_pro01/")
    print("  best.pt: runs_fifa/pilot_pro01/weights/best.pt")
    print("  학습곡선: runs_fifa/pilot_pro01/results.png")


if __name__ == "__main__":
    main()

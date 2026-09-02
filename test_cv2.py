"""
cv2 GUI 진단 스크립트
어디서 멈추는지 단계별로 확인. 모든 print는 flush=True 로 즉시 출력.
"""
import sys

print("STEP 1: Python 시작", flush=True)
print(f"        Python {sys.version}", flush=True)

print("STEP 2: cv2 임포트 시도...", flush=True)
import cv2
print(f"        cv2 임포트 OK, 버전 {cv2.__version__}", flush=True)

print("STEP 3: numpy 임포트 + 더미 이미지 생성", flush=True)
import numpy as np
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (0, 200, 0)  # 녹색 박스
cv2.putText(img, "If you see this, GUI works!", (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
print("        이미지 생성 OK", flush=True)

print("STEP 4: namedWindow + setWindowProperty TOPMOST", flush=True)
win = "cv2 test"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
print("        창 생성 호출 완료", flush=True)

print("STEP 5: imshow 호출", flush=True)
cv2.imshow(win, img)
print("        imshow 완료 → 창이 보여야 함", flush=True)

print("STEP 6: waitKey(0) 진입 — 창이 보이면 아무 키나 눌러서 닫기", flush=True)
key = cv2.waitKey(0)
print(f"        waitKey 종료, 키코드={key}", flush=True)

cv2.destroyAllWindows()
print("STEP 7: 완료. 정상 종료.", flush=True)

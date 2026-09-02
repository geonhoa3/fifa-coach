"""
YOLO 기반 미니맵 dot 검출 — 색마스킹(detect_teams) 대체용 어댑터.

학습된 best.pt 로 미니맵에서 공/양팀/조작선수를 검출하고,
기존 extract_game_state 가 기대하는 형식(ball, player, t0_dots, t1_dots)으로 변환.

클래스 매핑 (Roboflow data.yaml 순서):
  0=ball, 1=team1, 2=team1_active, 3=team2, 4=team2_active

사용:
  from src.detect_yolo import YoloDetector
  det = YoloDetector()              # best.pt 자동 로드
  state = det.detect(mini_bgr)      # {'ball','player','t0_dots','t1_dots'}
"""
from pathlib import Path
import numpy as np

_RUNS = Path(__file__).resolve().parent.parent / "runs_fifa"
# 우선순위: 최신 2클래스(dot) 모델 → 구 5클래스 파일럿 → 부트캠프 폴더 fallback
_CANDIDATES = [
    _RUNS / "pro01_09_dot" / "weights" / "best.pt",
    _RUNS / "pro01_08_dot" / "weights" / "best.pt",
    _RUNS / "pro010203040506_dot" / "weights" / "best.pt",
    _RUNS / "pro0102030405_dot" / "weights" / "best.pt",
    _RUNS / "pro01020304_dot" / "weights" / "best.pt",
    _RUNS / "pro010203_dot" / "weights" / "best.pt",
    _RUNS / "pro0102_dot" / "weights" / "best.pt",
    _RUNS / "pilot_pro01" / "weights" / "best.pt",
    Path(r"C:\python\01_codeit_project01\pjt-sprint_ai07_healthcare\runs\detect\runs_fifa\pilot_pro01\weights\best.pt"),
]
_MODEL_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

# 클래스 id → 의미
BALL, TEAM1, TEAM1_ACT, TEAM2, TEAM2_ACT = 0, 1, 2, 3, 4


class YoloDetector:
    def __init__(self, model_path: str = None, conf: float = 0.25):
        from ultralytics import YOLO
        p = model_path or str(_MODEL_PATH)
        if not Path(p).exists():
            raise FileNotFoundError(f"YOLO 모델 없음: {p}  (먼저 train_yolo.py 학습)")
        self.model = YOLO(p)
        self.conf = conf

    def detect(self, mini_bgr):
        """미니맵 한 장 → game_state dict.
        t0=team1, t1=team2. active(조작선수)는 해당 팀 dot에 포함 + player로 별도 표시.
        """
        r = self.model.predict(mini_bgr, conf=self.conf, verbose=False)[0]
        ball, player, t0, t1 = None, None, [], []
        best_ball_area = 0
        for b in r.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if cls == BALL:
                area = (x2 - x1) * (y2 - y1)
                if area > best_ball_area:    # 가장 큰 공 하나만
                    best_ball_area = area; ball = (cx, cy)
            elif cls in (TEAM1, TEAM1_ACT):
                t0.append((cx, cy))
                if cls == TEAM1_ACT:
                    player = (cx, cy)
            elif cls in (TEAM2, TEAM2_ACT):
                t1.append((cx, cy))
                if cls == TEAM2_ACT:
                    player = (cx, cy)
        return {"ball": ball, "player": player, "t0_dots": t0, "t1_dots": t1, "valid": True}


class YoloDotDetector:
    """2클래스(ball/dot) 모델용 — dot 위치만 검출 후 색으로 팀 분리.
    색 무관 검출 + 사후 색분리 = 해상도·팀색 일반화. 클래스: 0=ball, 1=dot
    """
    def __init__(self, model_path: str = None, conf: float = 0.25):
        from ultralytics import YOLO
        p = model_path or str(_MODEL_PATH)
        if not Path(p).exists():
            raise FileNotFoundError(f"YOLO 모델 없음: {p}")
        self.model = YOLO(p)
        self.conf = conf
        self.ref_colors = None   # (c0, c1) BGR — init_ref_colors 로 고정

    def init_ref_colors(self, video_path: str, roi, n_attempts: int = 12):
        """영상 여러 시점에서 dot 색을 모아 K-Means(k=2)로 팀 참조색 고정.

        이후 detect()는 매 프레임 참조색과의 거리로 배정 →
        프레임 간 T0/T1 라벨 일관성 보장 (스왑 방지).
        """
        import cv2
        from src.minimap import crop_minimap
        from src.detect_teams import is_minimap_valid, has_field_rect
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        colors = []
        for i in range(n_attempts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 1) / (n_attempts + 1)))
            ok, f = cap.read()
            if not ok:
                continue
            mini = crop_minimap(f, roi)
            if mini.size == 0 or not is_minimap_valid(mini) or not has_field_rect(mini):
                continue
            r = self.model.predict(mini, conf=self.conf, verbose=False)[0]
            for b in r.boxes:
                if int(b.cls[0]) != 0:   # dot만
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    colors.append(self._dot_color(
                        mini, int((x1 + x2) / 2), int((y1 + y2) / 2)))
        cap.release()
        if len(colors) < 8:
            print("[경고] 참조색 표본 부족 → 프레임별 밝기 분리로 fallback")
            return None
        data = np.array(colors, dtype=np.float32)
        _, labels, centers = cv2.kmeans(
            data, 2, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
            5, cv2.KMEANS_PP_CENTERS)
        # V(밝기) 높은 쪽을 t0 으로 고정 (영상 무관 일관 규칙)
        v = centers.mean(axis=1)
        c0, c1 = (centers[0], centers[1]) if v[0] >= v[1] else (centers[1], centers[0])
        self.ref_colors = (c0, c1)
        print(f"[참조색 고정] T0 BGR={c0.round(0)}, T1 BGR={c1.round(0)} (표본 {len(colors)})")
        return self.ref_colors

    def _dot_color(self, mini_bgr, cx, cy, r=3):
        import cv2
        h, w = mini_bgr.shape[:2]
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        patch = mini_bgr[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch) == 0:
            return np.array([0, 0, 0])
        hsv = cv2.cvtColor(patch.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        nong = patch[~((hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 2] < 200))]
        return (nong if len(nong) else patch).mean(axis=0)

    def detect(self, mini_bgr):
        import cv2
        r = self.model.predict(mini_bgr, conf=self.conf, verbose=False)[0]
        ball, dots, best_ball = None, [], 0
        for b in r.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if cls == 0:
                a = (x2 - x1) * (y2 - y1)
                if a > best_ball:
                    best_ball = a; ball = (cx, cy)
            else:
                dots.append((cx, cy))
        t0, t1 = [], []
        if dots:
            cols = np.array([self._dot_color(mini_bgr, x, y) for x, y in dots])
            if self.ref_colors is not None:
                # 고정 참조색과의 거리로 배정 → 프레임 간 라벨 일관
                d0 = ((cols - self.ref_colors[0]) ** 2).sum(axis=1)
                d1 = ((cols - self.ref_colors[1]) ** 2).sum(axis=1)
                for (x, y), a, b_ in zip(dots, d0, d1):
                    (t0 if a <= b_ else t1).append((x, y))
            else:
                # fallback: 프레임별 밝기 중간값 분리 (라벨 스왑 가능성 있음)
                V = cv2.cvtColor(cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)[:, 2]
                thr = (float(V.max()) + float(V.min())) / 2 if len(V) > 1 else 128
                for (x, y), v in zip(dots, V):
                    (t0 if v >= thr else t1).append((x, y))
        return {"ball": ball, "player": None, "t0_dots": t0, "t1_dots": t1, "valid": True}


def compare_with_colormask(video_path: str, frame_idx: int):
    """디버그: 같은 프레임에서 색마스킹 vs YOLO 검출 수 비교."""
    import cv2
    from src.minimap import load_roi, crop_minimap
    from src.extract_game_state import init_team_model, extract_frame_state
    cap = cv2.VideoCapture(video_path); cap.set(1, frame_idx); ok, f = cap.read(); cap.release()
    mini = crop_minimap(f, load_roi(video_path=video_path))
    tm = init_team_model(video_path)
    cm = extract_frame_state(mini, tm)
    y = YoloDetector().detect(mini)
    print(f"[색마스킹] T0={len(cm['t0_dots'])} T1={len(cm['t1_dots'])} ball={cm['ball'] is not None}")
    print(f"[YOLO]    T0={len(y['t0_dots'])} T1={len(y['t1_dots'])} ball={y['ball'] is not None} player={y['player'] is not None}")


if __name__ == "__main__":
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "data/videos/pro01.mp4"
    fi = int(sys.argv[2]) if len(sys.argv) > 2 else 10800
    compare_with_colormask(v, fi)

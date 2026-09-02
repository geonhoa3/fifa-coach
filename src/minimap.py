"""
Step 1 — 미니맵 ROI 캘리브레이션

영상에서 미니맵 영역의 픽셀 좌표 (x, y, w, h)를 한 번 잡아두고
이후 모든 프레임에서 같은 영역을 잘라내는 데 사용한다.

핵심 아이디어:
- cv2.selectROI 가 사용자 마우스 드래그 → 좌표 반환을 다 해준다
- 좌표는 JSON 으로 저장 → 다음 실행부터는 캘리브레이션 생략
"""

import cv2
import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "minimap_roi.json"


def grab_frame(video_path: str, frame_idx: int = 300, verbose: bool = False):
    """영상에서 N번째 프레임 한 장 추출. verbose=False 면 진단 출력 생략."""
    import os

    abs_path = Path(video_path).resolve()
    if verbose:
        print(f"[진단] 작업 폴더: {os.getcwd()}")
        print(f"[진단] 입력 경로: {video_path}")
        print(f"[진단] 절대 경로: {abs_path}")
        print(f"[진단] 파일 존재: {abs_path.exists()}")
    if not abs_path.exists():
        raise FileNotFoundError(f"영상 파일이 없습니다: {abs_path}")

    cap = cv2.VideoCapture(str(abs_path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다 (코덱 문제 가능성): {abs_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print(f"[진단] 영상 정보: {w}x{h}, {fps:.1f}fps, 총 {n_frames}프레임")

    if frame_idx >= n_frames:
        frame_idx = n_frames // 2
        if verbose:
            print(f"[진단] frame_idx 조정: {frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"프레임 읽기 실패: {abs_path} @ {frame_idx}")
    if verbose:
        print(f"[진단] 프레임 읽기 성공 (shape={frame.shape})")
    return frame


def calibrate_roi(video_path: str, frame_idx: int = None) -> tuple[int, int, int, int]:
    """
    마우스 드래그로 미니맵 영역을 잡고 좌표 (x, y, w, h) 반환.
    창에서 ENTER 또는 SPACE 로 확정, c 키로 취소.
    frame_idx=None 이면 영상 중간 시점(게임 진행 중일 확률 높음) 사용.
    """
    if frame_idx is None:
        # 영상 중간 시점 자동 선택
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = n_frames // 2
    frame = grab_frame(video_path, frame_idx=frame_idx)

    win = "Drag minimap area, press ENTER"
    print(f"[진단] 창 생성 중: '{win}'")
    print("[안내] 창이 안 보이면 Alt+Tab 또는 작업표시줄 확인")

    # 명시적 namedWindow + 최상위 강제 → 다른 창 뒤에 숨는 것 방지
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow(win, frame)
    cv2.waitKey(500)  # 창 렌더링 대기

    roi = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        raise ValueError("ROI 가 선택되지 않았습니다.")
    return tuple(int(v) for v in roi)


def _video_key(video_path: str) -> str:
    """영상 경로 → manifest key (basename 확장자 제거)."""
    return Path(video_path).stem


def save_roi(roi: tuple, video_path: str = None) -> None:
    """영상별 ROI 저장. video_path 없으면 'default' 키로."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {"rois": {}}
    if CONFIG_PATH.exists():
        existing = json.loads(CONFIG_PATH.read_text())
        if "rois" in existing:
            data = existing
        elif "roi" in existing:
            # 옛 구조 자동 마이그레이션
            data["rois"]["default"] = list(existing["roi"])

    key = _video_key(video_path) if video_path else "default"
    data["rois"][key] = list(roi)
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[저장됨] {CONFIG_PATH} [{key}]: {roi}")


def load_roi(video_path: str = None) -> tuple[int, int, int, int]:
    """영상별 ROI 로드.
    - video_path 주면 해당 영상 키로 조회
    - 키 매칭 안 되거나 video_path 없으면 default 사용
    - 옛 구조 ({"roi": [...]}) 도 그대로 지원
    """
    data = json.loads(CONFIG_PATH.read_text())

    # 옛 구조 (backward compat)
    if "roi" in data and "rois" not in data:
        return tuple(data["roi"])

    rois = data.get("rois", {})
    if not rois:
        raise ValueError(f"{CONFIG_PATH} 에 ROI 가 없습니다. 먼저 calibrate 하세요.")

    if video_path:
        key = _video_key(video_path)
        if key in rois:
            return tuple(rois[key])
        if "default" in rois:
            print(f"[경고] {key} ROI 없음 → default 사용. "
                  f"권장: python -m src.minimap calibrate {video_path}")
            return tuple(rois["default"])
        # default도 없으면 첫 번째 사용
        first_key = next(iter(rois))
        print(f"[경고] {key} ROI 없음 → {first_key} ROI 차용. 부정확할 수 있음.")
        return tuple(rois[first_key])

    # video_path 없음 → default 또는 첫 번째
    if "default" in rois:
        return tuple(rois["default"])
    return tuple(next(iter(rois.values())))


def list_rois() -> dict:
    """저장된 영상별 ROI 목록."""
    if not CONFIG_PATH.exists():
        return {}
    data = json.loads(CONFIG_PATH.read_text())
    if "roi" in data and "rois" not in data:
        return {"default": data["roi"]}
    return data.get("rois", {})


def crop_minimap(frame, roi):
    """프레임에서 미니맵 영역만 잘라낸다."""
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


# ─────────────────────────────────────────────────────────────
# 사용 흐름:
#   1. 캘리브레이션 1회:   python -m src.minimap calibrate data/videos/sample.mp4
#   2. 결과 확인:          python -m src.minimap preview data/videos/sample.mp4
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    video = sys.argv[2] if len(sys.argv) > 2 else "data/videos/sample.mp4"

    # 3번째 인자로 frame_idx 지정 가능 (예: 3000)
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if cmd == "calibrate":
        roi = calibrate_roi(video, frame_idx=frame_idx)
        save_roi(roi, video_path=video)

    elif cmd == "preview":
        roi = load_roi(video_path=video)
        frame = grab_frame(video, frame_idx=frame_idx or 300)
        mini = crop_minimap(frame, roi)
        cv2.imshow("Minimap (q to quit)", mini)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif cmd == "list":
        rois = list_rois()
        if not rois:
            print("[비어 있음] 저장된 ROI 없음")
        else:
            print(f"[저장된 ROI {len(rois)}개]")
            for k, v in rois.items():
                print(f"  {k}: {v}")

    else:
        print("usage: python -m src.minimap [calibrate|preview|list] <video_path>")

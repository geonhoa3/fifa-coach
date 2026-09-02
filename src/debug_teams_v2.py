"""
v2 팀 분류 시각화 디버그 — classify_dots_v2 결과를 미니맵 위에 마커로 그린다.

목적: 4.8/9.5 같은 dot 카운트가 '진짜 팀 분리'인지 눈으로 확정.
  - 파랑 마커 = T0 (흰/회색 팀)
  - 빨강 마커 = T1 (색 팀)
  - 노랑 원   = 공
  - 자홍 십자 = 조작선수(빨강 dot, 분류에서 제외된 것)

실행:
  python -m src.debug_teams_v2 data/videos/match03.mp4 9000
  python -m src.debug_teams_v2 data/videos/match03.mp4 9000 save   # 창 대신 PNG 저장
"""
import sys
import cv2
import numpy as np

from src.minimap import load_roi, crop_minimap, grab_frame
from src.detect_ball import detect_ball
from src.detect_player import detect_red_dot
from src.extract_game_state import init_team_model
from src.detect_teams import classify_dots_v2


def render(video_path: str, frame_idx: int):
    model = init_team_model(video_path)
    roi = load_roi(video_path=video_path)
    frame = grab_frame(video_path, frame_idx=frame_idx)
    mini = crop_minimap(frame, roi)

    t0, t1 = classify_dots_v2(mini, model)
    ball = detect_ball(mini)
    player = detect_red_dot(mini)

    SCALE = 5
    h, w = mini.shape[:2]
    viz = cv2.resize(mini, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    def sc(p):
        return (p[0] * SCALE + SCALE // 2, p[1] * SCALE + SCALE // 2)

    for cx, cy in t0:   # T0 = 흰/회색 팀 → 파랑 사각
        cv2.drawMarker(viz, sc((cx, cy)), (255, 80, 0), cv2.MARKER_SQUARE, 16, 2)
    for cx, cy in t1:   # T1 = 색 팀 → 빨강 동그라미
        cv2.circle(viz, sc((cx, cy)), 8, (0, 0, 255), 2)
    if ball:
        cv2.circle(viz, sc(ball), 11, (0, 255, 255), 2)
    if player:
        cv2.drawMarker(viz, sc(player), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)

    txt = f"model={model['mode']}  T0(white/sq)={len(t0)}  T1(color/circ)={len(t1)}"
    cv2.putText(viz, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    print(f"[디버그] {txt}  ball={ball} player={player}")
    return viz


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/match03.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
    mode = sys.argv[3] if len(sys.argv) > 3 else "show"

    viz = render(video, frame_idx)
    if mode == "save":
        out = f"data/output/debug_teams_{frame_idx}.png"
        cv2.imwrite(out, viz)
        print(f"[저장] {out}")
    else:
        win = "v2 team classify (T0=blue square, T1=red circle | any key=close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(win, viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""
자동화 파이프라인 — 영상 여러 개를 일괄 처리.

각 영상마다:
  1. extract_game_state → motion → events → grid → xt_model → evaluate_actions
  2. 결과를 data/output/<영상명>/ 폴더로 자동 이동

실행:
  python -m src.process_all data/videos/match01.mp4 data/videos/match04.mp4
  python -m src.process_all data/videos/match*.mp4   # 전체
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# 자식 프로세스 stdout 을 UTF-8 로 강제 — 콘솔이 cp949 여도
# '—' 같은 특수문자 print 에서 UnicodeEncodeError 로 죽지 않게 함
SUBPROC_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


PIPELINE = [
    # (모듈명, 추가 인자)  — 영상 경로가 필요한 모듈은 VIDEO_ARG_MODULES 에
    ("src.extract_game_state", ["3"]),
    ("src.analyze_motion", []),
    ("src.detect_goals_ocr", []),   # 스코어보드 OCR → goals_ocr.json
    ("src.detect_events", []),      # goals_ocr.json 있으면 골 대조/정정
    ("src.grid", []),
    ("src.xt_model", ["--no-display"]),
    ("src.evaluate_actions", []),
]

VIDEO_ARG_MODULES = {"src.extract_game_state", "src.detect_goals_ocr"}


def process(video_path: str, observer: bool = False, use_yolo: bool = False) -> bool:
    name = Path(video_path).stem
    out_dir = Path("data/output") / name
    print(f"\n{'=' * 70}")
    print(f"[영상 처리] {video_path}  →  {out_dir}/"
          + ("  (관전 모드)" if observer else ""))
    print("=" * 70)

    # 출력 디렉토리 준비
    out_dir.mkdir(parents=True, exist_ok=True)

    # 파이프라인 실행
    for module, extra in PIPELINE:
        # --observer 플래그는 detect_events 에 전달
        # --strict-valid 는 항상 적용: 골 리플레이는 ROI 에 잔디가 보여
        # 잔디 비율 검사만으로는 valid 통과 → dot 개수 검사로 걸러냄
        # (match08: 골 3개 전부 리플레이 오염 데이터로 놓친 사고 재발 방지)
        extra_flags = list(extra)
        if observer and module == "src.detect_events":
            extra_flags.append("--observer")
        if module == "src.extract_game_state":
            extra_flags.append("--strict-valid")
        if use_yolo and module == "src.extract_game_state":
            extra_flags.append("--yolo")

        if module in VIDEO_ARG_MODULES:
            cmd = ["python", "-m", module, video_path] + extra_flags
        else:
            cmd = ["python", "-m", module] + extra_flags
        print(f"\n  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, env=SUBPROC_ENV)
        if result.returncode != 0:
            if module == "src.detect_goals_ocr":
                # OCR 은 보조 단계 — 실패해도 휴리스틱 골 추론으로 진행
                print(f"  [경고] {module} 실패 — OCR 없이 계속")
                continue
            print(f"  [실패] {module} (returncode {result.returncode})")
            return False

    # 결과 이동
    tmp = Path("data/output")
    moved = 0
    for pattern in ["*.csv", "*.json", "*.png"]:
        for f in tmp.glob(pattern):
            if f.parent == tmp:  # 하위 폴더는 건너뜀
                dst = out_dir / f.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(f), str(dst))
                moved += 1
    print(f"\n  [완료] 결과 파일 {moved}개 → {out_dir}/")
    return True


def main():
    args = sys.argv[1:]
    if not args:
        print("사용법: python -m src.process_all <video1> [video2] ... [옵션]")
        print("  --no-track  : 처리 후 track.py 자동 호출 비활성화 (기본은 호출)")
        print("  --pro       : 고수(V_pro) 학습용 영상으로 표시 (track 호출 안 함)")
        print("  --observer  : 관전 모드 영상 (양팀 다 색 dot). 양팀 분석 → V_pro")
        sys.exit(1)

    no_track = "--no-track" in args
    is_pro = "--pro" in args
    observer = "--observer" in args
    use_yolo = "--yolo" in args
    videos = [a for a in args if not a.startswith("--")]

    success = []
    fail = []
    for v in videos:
        if process(v, observer=observer, use_yolo=use_yolo):
            success.append(v)
        else:
            fail.append(v)

    print(f"\n{'=' * 70}")
    print(f"[전체 완료] 성공 {len(success)} / 실패 {len(fail)}")
    if fail:
        print(f"  실패한 영상: {fail}")

    # 자동 후크 — 사용자 영상이면 track.py 호출
    if success and not no_track and not is_pro:
        print(f"\n{'=' * 70}")
        print(f"[자동 후크] track.py 호출 — 코칭 카드 갱신")
        print("=" * 70)
        video_names = [Path(v).stem for v in success]
        cmd = ["python", "-m", "src.track"] + video_names
        print(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, env=SUBPROC_ENV)
    elif is_pro:
        print(f"\n[안내] --pro 플래그 → track 호출 안 함. "
              f"V_pro 생성하려면:")
        print(f"  python -m src.combine_xt {' '.join(Path(v).stem for v in success)} --output V_pro")


if __name__ == "__main__":
    main()

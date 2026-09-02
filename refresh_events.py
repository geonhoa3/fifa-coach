"""
이벤트 재평가 배치 — YOLO 추론 없이 저장된 motion.csv 로부터
detect_events → grid → xt_model → evaluate_actions 만 다시 실행.

용도: detect_events/evaluate_actions 로직 수정 후 (순간이동 필터,
무데이터 셀 제외 등) 전체 영상의 events/xt 결과를 빠르게 갱신.
영상당 몇 초 (CSV 연산만).

실행:
  python refresh_events.py pro01 pro02 ... pro09 --observer
  python refresh_events.py --all --observer   # data/output/ 하위 전체
"""
import sys
import shutil
import subprocess
from pathlib import Path

OUT = Path("data/output")
NEEDED = ["game_state.csv", "game_state.meta.json", "motion.csv"]
MODULES = [
    ("src.detect_events", True),   # (모듈, --observer 전달 여부)
    ("src.grid", False),
    ("src.xt_model", False),
    ("src.evaluate_actions", False),
]


def refresh(name: str, observer: bool) -> bool:
    src_dir = OUT / name
    if not all((src_dir / f).exists() for f in NEEDED):
        print(f"[{name}] 필요 파일 없음 → 스킵")
        return False
    print(f"\n{'=' * 60}\n[갱신] {name}\n{'=' * 60}")

    # 저장본을 루트로 복사
    for f in NEEDED:
        shutil.copy(src_dir / f, OUT / f)

    # 구버전 meta 에 roi 가 없으면 minimap_roi.json 에서 주입
    # (grid.py 가 영상별 미니맵 크기를 정확히 쓰도록)
    import json
    meta_p = OUT / "game_state.meta.json"
    meta = json.loads(meta_p.read_text())
    if "roi" not in meta:
        rois = json.loads(Path("data/minimap_roi.json").read_text())["rois"]
        if name in rois:
            meta["roi"] = rois[name]
            meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f"[{name}] meta 에 roi 주입: {rois[name]}")

    for module, pass_observer in MODULES:
        cmd = ["python", "-m", module]
        if module == "src.xt_model":
            cmd.append("--no-display")
        if pass_observer and observer:
            cmd.append("--observer")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[{name}] {module} 실패")
            return False

    # 결과를 영상 폴더로 되돌림
    for f in ["events.csv", "xt_values.json", "xt_heatmap.png", "grid_meta.json"]:
        p = OUT / f
        if p.exists():
            shutil.move(str(p), str(src_dir / f))
    # 루트에 복사했던 입력 정리
    for f in NEEDED:
        (OUT / f).unlink(missing_ok=True)
    print(f"[{name}] 완료 → {src_dir}/")
    return True


def main():
    args = sys.argv[1:]
    observer = "--observer" in args
    names = [a for a in args if not a.startswith("--")]
    if "--all" in args:
        names = sorted(d.name for d in OUT.iterdir()
                       if d.is_dir() and d.name != "combined"
                       and (d / "motion.csv").exists())
    if not names:
        print(__doc__)
        sys.exit(1)

    ok = [n for n in names if refresh(n, observer)]
    print(f"\n[전체 완료] {len(ok)}/{len(names)}")
    if ok:
        print(f"V_pro 재생성:\n  python -m src.combine_xt {' '.join(ok)} --output V_pro")


if __name__ == "__main__":
    main()

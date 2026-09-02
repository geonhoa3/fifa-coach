"""
통계지 OCR — 통계지 이미지 → ground_truth.json

흐름:
  1. EasyOCR 로 모든 텍스트 + bbox 추출
  2. 큰 글씨(높이 임계값) → 점수
  3. 라벨(슛, 패스성공률 등) 같은 row 에서 좌/우 숫자 매칭

실행:
  python -m src.parse_stats data/test/stats_match02.png
"""

import easyocr
import json
import re
from pathlib import Path


# 통계지 위→아래 순서 (게임 화면 고정 순서)
LABELS_ORDER = [
    "shots",            # 슛
    "shots_on_target",  # 유효슛
    "shot_accuracy",    # 슛 성공률 (%)
    "pass_accuracy",    # 패스 성공률 (%)
    "possession",       # 점유율 (%)
    "corner_kicks",     # 코너킥
    "tackles",          # 태클
    "fouls",            # 파울
    "offsides",         # 오프사이드
]

# 퍼센트 항목 (후처리 시 0~100 클리핑)
PERCENT_LABELS = {"shot_accuracy", "pass_accuracy", "possession"}


def parse_number(text: str):
    """텍스트에서 숫자만 추출 ('%' 떼고 정수)."""
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def postprocess(value, label_key: str):
    """% 항목은 OCR 오류 보정. 100 초과면 //10 (예: 259 → 25)."""
    if value is None:
        return None
    if label_key in PERCENT_LABELS and value > 100:
        value = value // 10
    return value


def group_rows(items, y_tolerance: float = 25):
    """items 를 y좌표 비슷한 것끼리 row 로 묶는다."""
    sorted_items = sorted(items, key=lambda i: i["y"])
    rows = []
    for item in sorted_items:
        if not rows or abs(item["y"] - rows[-1][0]["y"]) > y_tolerance:
            rows.append([item])
        else:
            rows[-1].append(item)
    return rows


def parse_stats(image_path: str, debug: bool = False) -> dict:
    print(f"[OCR] 이미지 로딩: {image_path}")
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    raw = reader.readtext(image_path)
    print(f"[OCR] 검출된 텍스트 {len(raw)}개")

    # 정규화: x_center, y_center, height
    items = []
    for bbox, text, conf in raw:
        x_center = sum(p[0] for p in bbox) / 4
        y_center = sum(p[1] for p in bbox) / 4
        height = abs(bbox[2][1] - bbox[0][1])
        items.append({
            "text": text.strip(),
            "x": x_center,
            "y": y_center,
            "h": height,
            "conf": conf,
        })

    if debug:
        print("\n[디버그] 모든 OCR 결과:")
        for it in sorted(items, key=lambda i: (i["y"], i["x"])):
            print(f"  y={it['y']:.0f} x={it['x']:.0f} h={it['h']:.0f} "
                  f"conf={it['conf']:.2f}  '{it['text']}'")

    # 점수: 가장 큰 글씨 2개
    score_candidates = sorted(
        [it for it in items if parse_number(it["text"]) is not None],
        key=lambda i: i["h"], reverse=True,
    )[:2]
    score_candidates.sort(key=lambda c: c["x"])
    score = {"us": None, "opp": None}
    if len(score_candidates) == 2:
        score["us"] = parse_number(score_candidates[0]["text"])
        score["opp"] = parse_number(score_candidates[1]["text"])

    print(f"[점수] {score}")

    # y 좌표 순서로 데이터 row 추적 (라벨 OCR 실패에 robust)
    rows = group_rows(items)

    # 데이터 row 필터: 점수 row(y=346) 와 HOME/AWAY(y=478) 제외
    # 좌측 숫자(x<1100) + 우측 숫자(x>1400) 둘 다 있는 row만
    data_rows = []
    for row in rows:
        y_mean = sum(it["y"] for it in row) / len(row)
        if y_mean < 500:
            continue
        left_nums = [it for it in row
                     if it["x"] < 1200 and parse_number(it["text"]) is not None]
        right_nums = [it for it in row
                      if it["x"] > 1400 and parse_number(it["text"]) is not None]
        if not left_nums or not right_nums:
            continue
        # 좌측은 가장 우측에 있는 숫자, 우측은 가장 좌측에 있는 숫자 (라벨 옆 숫자)
        left_nums.sort(key=lambda i: i["x"], reverse=True)
        right_nums.sort(key=lambda i: i["x"])
        data_rows.append({
            "y": y_mean,
            "us_text": left_nums[0]["text"],
            "opp_text": right_nums[0]["text"],
            "us": parse_number(left_nums[0]["text"]),
            "opp": parse_number(right_nums[0]["text"]),
        })

    # row 간 gap 의 중앙값 추정 (안정적)
    if len(data_rows) >= 3:
        gaps = [data_rows[i + 1]["y"] - data_rows[i]["y"]
                for i in range(len(data_rows) - 1)]
        gaps.sort()
        median_gap = gaps[len(gaps) // 2]
    else:
        median_gap = 57  # 기본값

    print(f"\n[row gap 중앙값] {median_gap:.0f}px")

    # 누락 row 검출: gap 이 median * 1.5 보다 크면 1개 이상 누락
    expanded = []
    for i, r in enumerate(data_rows):
        if i == 0:
            expanded.append(r)
            continue
        gap = r["y"] - data_rows[i - 1]["y"]
        n_missing = max(0, round(gap / median_gap) - 1)
        for _ in range(n_missing):
            expanded.append(None)
        expanded.append(r)

    print(f"\n[확장된 row {len(expanded)}개] (None = 누락)")
    for i, r in enumerate(expanded):
        marker = LABELS_ORDER[i] if i < len(LABELS_ORDER) else "(extra)"
        if r is None:
            print(f"  {i+1:2d}. (누락)              → {marker}")
        else:
            print(f"  {i+1:2d}. y={r['y']:.0f} | "
                  f"us='{r['us_text']}'({r['us']}) | "
                  f"opp='{r['opp_text']}'({r['opp']}) → {marker}")

    # LABELS_ORDER 순서대로 매핑 + 후처리
    stats = {"score": score}
    for i, key in enumerate(LABELS_ORDER):
        if i < len(expanded) and expanded[i] is not None:
            r = expanded[i]
            us = postprocess(r["us"], key)
            opp = postprocess(r["opp"], key)
            stats[key] = {"us": us, "opp": opp}
        else:
            stats[key] = {"us": None, "opp": None}

    return stats


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "data/test/stats_match02.png"
    debug = "--debug" in sys.argv

    stats = parse_stats(img, debug=debug)

    print("\n[최종 결과]")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    out = "data/output/ground_truth.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"\n[저장] {out}")

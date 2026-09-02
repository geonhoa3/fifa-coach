"""
자동라벨 4클래스 → 2클래스(dot/ball) 변환.

결정: YOLO는 'dot/ball'만 검출(색 무관 일반화), 팀 구분은 사후 색분리.
- 기존 클래스 0=ball → 0=ball (유지)
- 기존 1=team1, 3=team2 → 1=dot (합침)
- 기존 2/4=active → 1=dot (합침, active는 빨강마스킹으로 사후처리)

실행: python convert_labels_2class.py
"""
import glob, os

LBL_DIR = "data/yolo_v2/labels"
# old class id → new class id
MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}   # ball=0, 나머지 다 dot=1


def main():
    n = 0
    for p in glob.glob(f"{LBL_DIR}/*.txt"):
        if os.path.basename(p).startswith("_"):
            continue
        out = []
        for line in open(p):
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            out.append(f"{MAP.get(c, 1)} {' '.join(parts[1:])}")
        open(p, "w").write("\n".join(out) + "\n")
        n += 1
    open(f"{LBL_DIR}/_classes.txt", "w").write("ball\ndot\n")
    print(f"[변환] {n}개 라벨 → 2클래스 (0=ball, 1=dot)")


if __name__ == "__main__":
    main()

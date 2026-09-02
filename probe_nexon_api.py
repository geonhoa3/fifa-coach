# -*- coding: utf-8 -*-
"""
넥슨 FC 온라인 Open API 탐침 — 핵심 질문: 골 시각(goalTime)이 나오는가?

이게 나오면 영상 CV 없이도 「골 넣은 직후 실점」 진단이 가능하다.

사용법:
  1) https://openapi.nexon.com 에서 API 키 발급
  2) 키를 환경변수에 등록 (본인 터미널에서 직접 실행, 키는 이 파일에 넣지 말 것)
       setx NEXON_API_KEY "발급받은키"
     → 터미널 새로 열고
  3) python probe_nexon_api.py [닉네임]
"""
import os, sys, json, urllib.parse, urllib.request
from pathlib import Path

BASE = "https://open.api.nexon.com/fconline/v1"
NICK = sys.argv[1] if len(sys.argv) > 1 else "09이건호"

# 키 출처: 환경변수 → .nexon_key 파일 (둘 다 이 스크립트가 값을 출력하지 않음)
KEY = os.environ.get("NEXON_API_KEY")
if not KEY:
    kf = Path(__file__).with_name(".nexon_key")
    if kf.exists():
        KEY = kf.read_text(encoding="utf-8").strip()

if not KEY:
    print("[중단] API 키를 찾지 못했습니다. 둘 중 하나로 넣어주세요:")
    print('  A) setx NEXON_API_KEY "발급받은키"  → 새 터미널에서 실행')
    print("  B) 이 폴더에 .nexon_key 파일 생성 후 키만 한 줄로 저장")
    sys.exit(1)


def get(path, **params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-nxopen-api-key": KEY})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        print(f"[HTTP {e.code}] {path} → {body}")
        return None


def walk_keys(obj, prefix="", depth=0, out=None):
    """중첩 구조의 키 경로를 수집."""
    if out is None: out = []
    if depth > 3: return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                walk_keys(v, p, depth + 1, out)
            else:
                out.append((p, v))
    elif isinstance(obj, list) and obj:
        walk_keys(obj[0], prefix + "[0]", depth + 1, out)
    return out


print(f"[1] 닉네임 → ouid  ({NICK})")
uid = get("id", nickname=NICK)
if not uid: sys.exit(1)
ouid = uid.get("ouid")
print(f"    ouid = {ouid[:8]}…\n")

print("[2] 최근 공식경기(matchtype=50) 매치 목록")
ids = get("user/match", ouid=ouid, matchtype=50, offset=0, limit=5)
if not ids:
    sys.exit(1)
print(f"    매치 {len(ids)}개: {ids}\n")

print("[3] 가장 최근 매치 상세")
d = get("match-detail", matchid=ids[0])
if not d: sys.exit(1)

print(f"    최상위 키: {list(d.keys())}")
print(f"    matchDate: {d.get('matchDate')}  matchType: {d.get('matchType')}\n")

info = d.get("matchInfo", [])
print(f"[4] matchInfo {len(info)}명")
for p in info:
    md = p.get("matchDetail", {})
    sh = p.get("shoot", {})
    print(f"    - {p.get('nickname')}: 결과={md.get('matchResult')} "
          f"골={sh.get('goalTotal')} 슛={sh.get('shootTotal')} 점유율={md.get('possession')}")

print("\n[5] ★핵심★ shootDetail 필드 — 골 시각이 있는가?")
me = next((p for p in info if p.get("nickname") == NICK), info[0] if info else None)
sd = (me or {}).get("shootDetail", [])
if not sd:
    print("    shootDetail 없음 (또는 슛 기록 없음)")
else:
    print(f"    shootDetail {len(sd)}건. 첫 항목 전체:")
    print("    " + json.dumps(sd[0], ensure_ascii=False, indent=6).replace("\n", "\n    "))
    keys = set().union(*[set(x.keys()) for x in sd])
    print(f"\n    전체 필드: {sorted(keys)}")
    time_like = [k for k in keys if "time" in k.lower() or "min" in k.lower()]
    print(f"    ⏱ 시간 관련 필드: {time_like if time_like else '없음 ← CV 계속 필요'}")
    goals = [x for x in sd if x.get("goal")]
    print(f"    골 {len(goals)}건:")
    for g in goals:
        t = {k: g.get(k) for k in time_like}
        print(f"      goal=True  {t}  x={g.get('x')} y={g.get('y')} result={g.get('result')}")

print("\n[6] 선수별 status 키(참고)")
pl = (me or {}).get("player", [])
if pl:
    print(f"    {sorted(pl[0].get('status', {}).keys())}")

print("\n=== 판정 가이드 ===")
print("  시간 필드 있음 → 영상 CV 버리고 API로 전환 가능 (진단 자동화)")
print("  시간 필드 없음 → 골 '시각'은 스코어보드 OCR 유지, 나머지 집계는 API로 대체")

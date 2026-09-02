# -*- coding: utf-8 -*-
"""넥슨 FC 온라인 API 수집기 — 매치 상세를 로컬 캐싱.

사용:  python -m src.fetch_api            (기본: 공식친선 60)
       python -m src.fetch_api 60 50 40   (매치타입 지정)
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

BASE = "https://open.api.nexon.com/fconline/v1"
NICK = "09이건호"
CACHE = Path("data/api_cache")
KEYF = Path(".nexon_key")


def _key():
    import os
    k = os.environ.get("NEXON_API_KEY")
    if not k and KEYF.exists():
        k = KEYF.read_text(encoding="utf-8").strip()
    if not k:
        raise SystemExit("[중단] API 키 없음 (.nexon_key 또는 NEXON_API_KEY)")
    return k


KEY = _key()


def get(path, **params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"x-nxopen-api-key": KEY})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:            # rate limit → 백오프
                time.sleep(2 * (attempt + 1)); continue
            print(f"  [HTTP {e.code}] {path}")
            return None
        except Exception:
            time.sleep(1); continue
    return None


def match_detail(mid):
    """캐시 우선. 매치 상세는 불변이므로 영구 캐싱 가능."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{mid}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = get("match-detail", matchid=mid)
    if d:
        f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def collect(matchtypes=(60,), max_per_type=300):
    ouid = get("id", nickname=NICK)["ouid"]
    all_ids = {}
    for mt in matchtypes:
        ids = []
        for off in range(0, max_per_type, 50):
            batch = get("user/match", ouid=ouid, matchtype=mt, offset=off, limit=50)
            if not batch:
                break
            ids += batch
            if len(batch) < 50:
                break
        all_ids[mt] = ids
        print(f"[matchtype {mt}] 매치 {len(ids)}개")
        new = 0
        for i, mid in enumerate(ids):
            if not (CACHE / f"{mid}.json").exists():
                new += 1
            match_detail(mid)
            if (i + 1) % 25 == 0:
                print(f"   {i+1}/{len(ids)} …")
        print(f"   신규 {new}개 캐싱, 총 {len(ids)}개 확보")
    return all_ids


if __name__ == "__main__":
    mts = [int(x) for x in sys.argv[1:]] or [60]
    got = collect(tuple(mts))
    Path("data/api_cache/_index.json").write_text(
        json.dumps({str(k): v for k, v in got.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n[저장] data/api_cache/_index.json")

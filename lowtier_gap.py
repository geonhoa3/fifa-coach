# -*- coding: utf-8 -*-
"""하위 유저 메타 갭 검증 — 대상 하위 유저 vs 메타 기준선 vs 건호(근메타)."""
import json, time, urllib.parse, urllib.request
from pathlib import Path
from statistics import mean

KEY = Path('.nexon_key').read_text().strip()
CACHE = Path('data/api_cache_lowtier'); CACHE.mkdir(exist_ok=True)
def get(p, **q):
    u = f'https://open.api.nexon.com/fconline/v1/{p}?' + urllib.parse.urlencode(q)
    r = urllib.request.Request(u, headers={'x-nxopen-api-key': KEY})
    for _ in range(3):
        try: return json.loads(urllib.request.urlopen(r, timeout=20).read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(3); continue
            return None
        except: time.sleep(1)
    return None
def detail(mid):
    f = CACHE / f'{mid}.json'
    if f.exists(): return json.loads(f.read_text(encoding='utf-8'))
    d = get('match-detail', matchid=mid)
    if d: f.write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')
    return d

def n(v): return v or 0
def prof(p):
    md = p['matchDetail']; ps = p['pass']; sh = p['shoot']; df = p.get('defence', {})
    sd = [s for s in (p.get('shootDetail') or []) if s.get('x') is not None]
    if n(ps.get('passTry')) < 10 or not sd: return None
    return {
        '박스안슛%': 100*sum(1 for s in sd if s.get('inPenalty'))/len(sd),
        '골전환율%': 100*n(sh.get('goalTotal'))/max(1, n(sh.get('shootTotal'))),
        '유효슛%':   100*n(sh.get('effectiveShootTotal'))/max(1, n(sh.get('shootTotal'))),
        '패스성공%': 100*n(ps.get('passSuccess'))/max(1, n(ps.get('passTry'))),
        '드리블성공%': 100*_dsucc(p)/max(1, _dtry(p)),
        '태클성공%': 100*n(df.get('tackleSuccess'))/max(1, n(df.get('tackleTry'))),
        '점유율%':   n(md.get('possession')),
        '슛':        n(sh.get('shootTotal')),
    }
def _dsucc(p):
    return sum(n(x['status'].get('dribbleSuccess')) for x in p.get('player', []))
def _dtry(p):
    return sum(n(x['status'].get('dribbleTry')) for x in p.get('player', []))

def profile_user(nick, folders, maxg=25):
    """지정 폴더들에서 nick 의 경기 프로필 평균."""
    out = []
    for folder in folders:
        for f in Path(folder).glob('*.json'):
            if f.name.startswith('_'): continue
            d = json.loads(f.read_text(encoding='utf-8'))
            for pp in d.get('matchInfo', []):
                if pp['nickname'] == nick:
                    q = prof(pp)
                    if q: out.append(q)
    return out

def fetch_user(nick):
    d = get('id', nickname=nick)
    if not d: return
    ou = d['ouid']
    for mt in (60, 50):
        for off in (0,):
            ids = get('user/match', ouid=ou, matchtype=mt, offset=off, limit=25) or []
            for mid in ids: detail(mid)

# 분석할 하위등급 유저 닉네임 리스트 (예: ['nick1', 'nick2', ...])
TARGETS = []
for t in TARGETS:
    fetch_user(t)

keys = ['박스안슛%','골전환율%','유효슛%','패스성공%','드리블성공%','태클성공%','점유율%','슛']
# 메타 기준선
who = json.loads(Path('data/api_cache_elite/_who.json').read_text(encoding='utf-8'))
elite = profile_user.__wrapped__ if False else None
def avg_profile(profs):
    return {k: mean(x[k] for x in profs) for k in keys} if profs else None
elite_p = [prof(pp) for f in Path('data/api_cache_elite').glob('*.json') if not f.name.startswith('_')
           for pp in json.loads(f.read_text(encoding='utf-8')).get('matchInfo', []) if pp['nickname'] in who]
elite_p = avg_profile([x for x in elite_p if x])
gunho = avg_profile(profile_user('09이건호', ['data/api_cache']))
print(f"\n메타 기준선: 박스안슛 {elite_p['박스안슛%']:.0f}%  전환율 {elite_p['골전환율%']:.0f}%  패스 {elite_p['패스성공%']:.0f}%  점유 {elite_p['점유율%']:.0f}%\n")
print(f"{'유저':>14} {'판수':>4} {'박스안슛':>7} {'전환율':>6} {'패스%':>6} {'점유':>5}   판정")
gap_users = 0; usable = 0
for t in TARGETS:
    profs = profile_user(t, ['data/api_cache_lowtier'])
    if len(profs) < 3:
        print(f"{t:>14} {len(profs):>4}   ---데이터 부족---")
        continue
    usable += 1
    p = avg_profile(profs)
    # 메타보다 확실히 낮은 항목 개수 (5%p+ 차이)
    gaps = []
    if p['골전환율%'] < elite_p['골전환율%']-5: gaps.append('전환율↓')
    if p['패스성공%'] < elite_p['패스성공%']-5: gaps.append('패스↓')
    if p['박스안슛%'] < elite_p['박스안슛%']-8: gaps.append('원거리슛↑')
    if p['점유율%'] < elite_p['점유율%']-5: gaps.append('점유↓')
    verdict = '갭 뚜렷: '+','.join(gaps) if gaps else '메타 수준 (갭 거의 없음)'
    if gaps: gap_users += 1
    print(f"{t:>14} {len(profs):>4} {p['박스안슛%']:>6.0f}% {p['골전환율%']:>5.0f}% {p['패스성공%']:>5.0f}% {p['점유율%']:>4.0f}%   {verdict}")
print(f"\n=== 요약: 분석가능 {usable}명 중 뚜렷한 갭 {gap_users}명 ===")

# -*- coding: utf-8 -*-
"""메타 갭 리포트 시제품 — 대상 유저 vs 상위등급(메타) 기준선."""
import json
from pathlib import Path
from statistics import mean

def n(v): return v or 0
def prof(p):
    md = p['matchDetail']; ps = p['pass']; sh = p['shoot']; df = p.get('defence', {})
    sd = [s for s in (p.get('shootDetail') or []) if s.get('x') is not None]
    if n(ps.get('passTry')) < 10 or not sd: return None
    return {
        '박스안슛비중': 100*sum(1 for s in sd if s.get('inPenalty'))/len(sd),
        '골전환율':     100*n(sh.get('goalTotal'))/max(1, n(sh.get('shootTotal'))),
        '유효슛비율':   100*n(sh.get('effectiveShootTotal'))/max(1, n(sh.get('shootTotal'))),
        '패스성공률':   100*n(ps.get('passSuccess'))/max(1, n(ps.get('passTry'))),
        '스루패스비중': 100*n(ps.get('throughPassTry'))/max(1, n(ps.get('passTry'))),
        '드리블성공률': 100*n(md.get('dribble') and 1) if False else 100*n(sh.get('goalTotal') and 1),  # placeholder
        '점유율':       n(md.get('possession')),
        '슛':           n(sh.get('shootTotal')),
    }

def collect(folder, only=None):
    out = []
    for f in Path(folder).glob('*.json'):
        if f.name.startswith('_'): continue
        d = json.loads(f.read_text(encoding='utf-8'))
        for p in d.get('matchInfo', []):
            if only and p['nickname'] not in only: continue
            if not only and p['nickname'] == '09이건호':
                pass
            q = prof(p)
            if q: out.append(q)
    return out

# 대상: 09이건호 (본인 캐시)
me = [prof(p) for f in Path('data/api_cache').glob('*.json') if not f.name.startswith('_')
      for p in json.loads(f.read_text(encoding='utf-8')).get('matchInfo', []) if p['nickname'] == '09이건호']
me = [x for x in me if x]
# 메타 기준선: 상위등급 7명
who = json.loads(Path('data/api_cache_elite/_who.json').read_text(encoding='utf-8'))
elite = [prof(p) for f in Path('data/api_cache_elite').glob('*.json') if not f.name.startswith('_')
         for p in json.loads(f.read_text(encoding='utf-8')).get('matchInfo', []) if p['nickname'] in who]
elite = [x for x in elite if x]

keys = ['박스안슛비중','골전환율','유효슛비율','패스성공률','스루패스비중','점유율','슛']
higher_better = set(keys)  # 전부 높을수록 메타에 가까움(대체로)
print(f"메타 갭 리포트 — 09이건호  (본인 {len(me)}판 vs 메타 상위등급 {len(elite)}판)\n")
print(f"{'지표':>12} {'나':>7} {'메타':>7} {'갭':>7}   판정")
for k in keys:
    a = mean(x[k] for x in me); b = mean(x[k] for x in elite); g = a-b
    if abs(g)/max(abs(b),1e-9) < 0.05: v='메타 수준 ✔'
    elif g < 0: v='메타보다 낮음 ▼'
    else: v='메타 이상 ▲'
    unit = '' if k in ('슛',) else '%'
    print(f"{k:>12} {a:>6.1f}{unit} {b:>6.1f}{unit} {g:>+6.1f}   {v}")

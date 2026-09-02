# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from statistics import mean
KEY = Path('.nexon_key').read_text().strip()
NICK = '09이건호'
def get(p, **q):
    u = f'https://open.api.nexon.com/fconline/v1/{p}?' + urllib.parse.urlencode(q)
    r = urllib.request.Request(u, headers={'x-nxopen-api-key': KEY})
    try: return json.loads(urllib.request.urlopen(r, timeout=20).read().decode())
    except urllib.error.HTTPError: return None
def n(v): return v or 0

def row(me, opp, date):
    md = me['matchDetail']; sh = me['shoot']; ps = me['pass']; osh = opp['shoot']
    gf = sh.get('goalTotal'); ga = osh.get('goalTotal')
    if gf is None or ga is None: return None
    return {'date': date, 'res': '승' if gf > ga else ('패' if gf < ga else '무'),
            'gf': gf, 'ga': ga, 'poss': n(md.get('possession')),
            'pacc': 100*n(ps.get('passSuccess'))/max(1, n(ps.get('passTry'))),
            'shoot': n(sh.get('shootTotal')), 'oshoot': n(osh.get('shootTotal')),
            'conv': 100*gf/max(1, n(sh.get('shootTotal'))),
            'odiv': opp.get('division')}

# 랭크(50) — API 직접
ouid = get('id', nickname=NICK)['ouid']
ranked = []
ids = get('user/match', ouid=ouid, matchtype=50, offset=0, limit=30) or []
for mid in ids:
    md = get('match-detail', matchid=mid)
    if not md or len(md.get('matchInfo', [])) < 2 or not md.get('matchDate'): continue
    me = next((p for p in md['matchInfo'] if p['nickname'] == NICK), None)
    opp = next((p for p in md['matchInfo'] if p['nickname'] != NICK), None)
    if not me or not opp: continue
    r = row(me, opp, datetime.fromisoformat(md['matchDate']))
    if r: ranked.append(r)

# 친선(60) — 캐시
friendly = []
for f in Path('data/api_cache').glob('*.json'):
    if f.name.startswith('_'): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    if len(d.get('matchInfo', [])) < 2 or not d.get('matchDate'): continue
    me = next((p for p in d['matchInfo'] if p['nickname'] == NICK), None)
    opp = next((p for p in d['matchInfo'] if p['nickname'] != NICK), None)
    if not me or not opp: continue
    r = row(me, opp, datetime.fromisoformat(d['matchDate']))
    if r: friendly.append(r)

def summ(gs, lab):
    w = sum(1 for x in gs if x['res']=='승'); dd=sum(1 for x in gs if x['res']=='무'); l=sum(1 for x in gs if x['res']=='패')
    divs = [x['odiv'] for x in gs if x['odiv']]
    print(f"[{lab}] {len(gs)}판  {w}승 {dd}무 {l}패 (승률 {100*w/len(gs):.0f}%)")
    print(f"   내 점유율 {mean(x['poss'] for x in gs):.1f}%  패스 {mean(x['pacc'] for x in gs):.1f}%  "
          f"내슛 {mean(x['shoot'] for x in gs):.1f}  전환 {mean(x['conv'] for x in gs):.0f}%")
    print(f"   상대슛 {mean(x['oshoot'] for x in gs):.1f}  실점 {mean(x['ga'] for x in gs):.2f}  상대평균등급 {mean(divs) if divs else 0:.0f}(낮을수록 강)")

summ(ranked, '랭크(공식경기)')
print()
summ(friendly, '친선(일반)')
print("\n[랭크 최근 순서]", ' '.join(x['res'] for x in sorted(ranked, key=lambda r: r['date'])))

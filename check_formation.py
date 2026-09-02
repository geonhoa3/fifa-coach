# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime, timedelta

KEY = Path('.nexon_key').read_text().strip()
def get(p, **q):
    u = f'https://open.api.nexon.com/fconline/v1/{p}?' + urllib.parse.urlencode(q)
    r = urllib.request.Request(u, headers={'x-nxopen-api-key': KEY})
    try: return json.loads(urllib.request.urlopen(r, timeout=20).read().decode())
    except urllib.error.HTTPError: return None

SUA, FOR = 845176580, 845049072
POS4222 = {26, 20}
DIV = {800:'슈챔',900:'챔피언스',1000:'슈챌',1100:'챌1',1200:'챌2',1300:'챌3',
       1900:'?1900',2000:'월클1',2100:'월클2',2200:'월클3',2300:'프로1',2400:'프로2',2500:'프로3',2600:'세미1'}

ouid = get('id', nickname='09이건호')['ouid']
ids = get('user/match', ouid=ouid, matchtype=60, offset=0, limit=50)

groups = {'4222': [], '42211': []}
mismatch = 0
for mid in ids:
    md = get('match-detail', matchid=mid)
    if not md or len(md.get('matchInfo', [])) < 2 or not md.get('matchDate'): continue
    me = next((p for p in md['matchInfo'] if p['nickname'] == '09이건호'), None)
    opp = next((p for p in md['matchInfo'] if p['nickname'] != '09이건호'), None)
    if not me or not opp: continue
    gf = me['shoot'].get('goalTotal'); ga = opp['shoot'].get('goalTotal')
    if gf is None or ga is None: continue          # 몰수 제외
    pl = {p['spId']: p for p in me.get('player', [])}
    if SUA not in pl or FOR not in pl: continue
    sp = {pl[SUA]['spPosition'], pl[FOR]['spPosition']}
    fmt = '4222' if sp <= POS4222 else '42211'
    res = '승' if gf > ga else ('패' if gf < ga else '무')     # 스코어 기준(확실)
    field_res = me['matchDetail'].get('matchResult')
    if field_res != res: mismatch += 1
    dt = (datetime.fromisoformat(md['matchDate']) + timedelta(hours=9)).strftime('%m-%d %H:%M')
    groups[fmt].append({'dt': dt, 'res': res, 'field': field_res, 'gf': gf, 'ga': ga,
                        'odiv': opp.get('division'), 'opp': opp['nickname'][:12]})

for fmt in ('4222', '42211'):
    gs = groups[fmt]
    if not gs: continue
    w = sum(1 for g in gs if g['res'] == '승'); d = sum(1 for g in gs if g['res'] == '무'); l = sum(1 for g in gs if g['res'] == '패')
    divs = [g['odiv'] for g in gs if g['odiv']]
    avgdiv = sum(divs)/len(divs) if divs else 0
    print(f"\n=== {fmt}: {w}승 {d}무 {l}패 ({len(gs)}판) | 득실 {sum(g['gf'] for g in gs)}:{sum(g['ga'] for g in gs)} | 상대평균등급 {avgdiv:.0f}(낮을수록 강) ===")
    for g in sorted(gs, key=lambda x: x['dt']):
        dv = DIV.get(g['odiv'], g['odiv'])
        note = '' if g['field'] == g['res'] else f"  (matchResult필드={g['field']}!)"
        print(f"  {g['dt']}  {g['gf']}:{g['ga']} {g['res']}  상대 {str(dv):>6}  {g['opp']}{note}")
print(f"\n스코어 vs matchResult필드 불일치: {mismatch}건")

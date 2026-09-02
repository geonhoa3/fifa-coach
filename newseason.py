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
def squad(p):
    g = [x['spGrade'] for x in p.get('player', []) if x.get('status', {}).get('spRating', 0) > 0]
    return mean(g) if g else 0

ouid = get('id', nickname=NICK)['ouid']
ids = get('user/match', ouid=ouid, matchtype=50, offset=0, limit=12) or []
print(f"최근 공식경기(랭크) — matchtype 50\n")
print(f"{'날짜(KST)':>13} {'스코어':>6} {'결과':>4} {'내스쿼드':>7} {'상대스쿼드':>8} {'내점유':>6} {'내슛':>4} {'상대슛':>5} {'상대등급':>7}")
rows = []
for mid in ids:
    md = get('match-detail', matchid=mid)
    if not md or len(md.get('matchInfo', [])) < 2 or not md.get('matchDate'): continue
    me = next((p for p in md['matchInfo'] if p['nickname'] == NICK), None)
    opp = next((p for p in md['matchInfo'] if p['nickname'] != NICK), None)
    if not me or not opp: continue
    gf = me['shoot'].get('goalTotal'); ga = opp['shoot'].get('goalTotal')
    if gf is None or ga is None:
        res = '몰수'
    else:
        res = '승' if gf > ga else ('패' if gf < ga else '무')
    kst = (datetime.fromisoformat(md['matchDate']) + timedelta(hours=9)).strftime('%m-%d %H:%M')
    md_ = me['matchDetail']
    print(f"{kst:>13} {str(gf)+':'+str(ga):>6} {res:>4} {squad(me):>7.2f} {squad(opp):>8.2f} "
          f"{n(md_.get('possession')):>5}% {n(me['shoot'].get('shootTotal')):>4} "
          f"{n(opp['shoot'].get('shootTotal')):>5} {str(opp.get('division')):>7}")

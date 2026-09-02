# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request, sys
from pathlib import Path
from datetime import datetime, timedelta
KEY = Path('.nexon_key').read_text().strip()
NICK = '09이건호'
OPP_NAME = sys.argv[1] if len(sys.argv) > 1 else 'OPPONENT_NICKNAME'  # 분석할 상대 닉네임 (인자로 전달)
POS = {0:'GK',3:'RB',4:'RCB',6:'LCB',7:'LB',9:'RDM',11:'LDM',12:'RM',16:'LM',
       20:'우ST',26:'좌ST',25:'ST',18:'CAM',24:'CF',17:'CAM',19:'CAM'}
def get(p, **q):
    u = f'https://open.api.nexon.com/fconline/v1/{p}?' + urllib.parse.urlencode(q)
    r = urllib.request.Request(u, headers={'x-nxopen-api-key': KEY})
    try: return json.loads(urllib.request.urlopen(r, timeout=20).read().decode())
    except urllib.error.HTTPError: return None
def n(v): return v or 0

ouid = get('id', nickname=NICK)['ouid']
found = None
for mt in (50, 60, 52):
    ids = get('user/match', ouid=ouid, matchtype=mt, offset=0, limit=20) or []
    for mid in ids:
        md = get('match-detail', matchid=mid)
        if not md or len(md.get('matchInfo', [])) < 2: continue
        opp = next((p for p in md['matchInfo'] if p['nickname'] != NICK), None)
        if opp and OPP_NAME in opp['nickname']:
            found = (mt, md); break
    if found: break

if not found:
    print(f"'{OPP_NAME}' 과의 경기가 아직 API에 없음 (2시간 지연 or 더 과거). 나중에 다시 시도.")
    sys.exit()

mt, md = found
me = next(p for p in md['matchInfo'] if p['nickname'] == NICK)
opp = next(p for p in md['matchInfo'] if p['nickname'] != NICK)
gf = me['shoot'].get('goalTotal'); ga = opp['shoot'].get('goalTotal')
kst = (datetime.fromisoformat(md['matchDate']) + timedelta(hours=9)).strftime('%m-%d %H:%M')
pl = {x['spId']: x for x in me.get('player', [])}
SUA, FOR = 845176580, 845049072
fmt = '?'
if SUA in pl and FOR in pl:
    sp = {pl[SUA]['spPosition'], pl[FOR]['spPosition']}
    fmt = '4222' if sp <= {26,20} else ('42211' if sp <= {25,18} else f'기타{sorted(sp)}')
print(f"=== {kst}  {gf}:{ga}  vs {opp['nickname']}  (matchtype {mt}, 포메 {fmt}) ===")
mdd = me['matchDetail']
print(f"점유율 {n(mdd.get('possession'))}%  내슛 {n(me['shoot'].get('shootTotal'))}(유효 {n(me['shoot'].get('effectiveShootTotal'))})  "
      f"상대슛 {n(opp['shoot'].get('shootTotal'))}(유효 {n(opp['shoot'].get('effectiveShootTotal'))})")
print(f"내 스쿼드 vs 상대: {sum(x['spGrade'] for x in me['player'] if x['status'].get('spRating',0)>0)/max(1,len([x for x in me['player'] if x['status'].get('spRating',0)>0])):.2f} "
      f"vs {sum(x['spGrade'] for x in opp['player'] if x['status'].get('spRating',0)>0)/max(1,len([x for x in opp['player'] if x['status'].get('spRating',0)>0])):.2f}   상대등급 {opp.get('division')}")

print("\n[내 선수 평점 (낮은 순 6)]")
rows = [(x['spPosition'], x['status']) for x in me.get('player', []) if x['status'].get('spRating', 0) > 0]
for pos, st in sorted(rows, key=lambda r: r[1]['spRating'])[:6]:
    print(f"  {POS.get(pos,pos):>5}  평점 {st['spRating']:.1f}  골{st.get('goal')} 슛{st.get('shoot')} "
          f"패스{st.get('passSuccess')}/{st.get('passTry')} 태클{st.get('tackle')}/{st.get('tackleTry')}")

half = lambda t: '후반' if (t>>24)&1 else '전반'
print(f"\n[상대가 {ga}골 넣은 경위]")
for s in opp.get('shootDetail', []):
    if s.get('result') == 3:
        loc = '박스안' if s.get('inPenalty') else '박스밖'
        print(f"  {half(s['goalTime'])} {(s['goalTime']&0xFFFFFF)//60}:{(s['goalTime']&0xFFFFFF)%60:02d}  {loc}  "
              f"x={s['x']:.2f} y={s['y']:.2f}  어시={s.get('assist')}")
print(f"\n[내 슛]")
for s in me.get('shootDetail', []):
    rr = {3:'골',2:'유효',1:'빗나감'}.get(s.get('result'), s.get('result'))
    loc = '박스안' if s.get('inPenalty') else '박스밖'
    print(f"  {rr}  {loc} x={s['x']:.2f} y={s['y']:.2f}")

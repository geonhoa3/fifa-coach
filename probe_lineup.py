# -*- coding: utf-8 -*-
import json, urllib.parse, urllib.request
from pathlib import Path
from datetime import datetime, timedelta

KEY = Path('.nexon_key').read_text().strip()

def get(p, **q):
    u = f'https://open.api.nexon.com/fconline/v1/{p}?' + urllib.parse.urlencode(q)
    r = urllib.request.Request(u, headers={'x-nxopen-api-key': KEY})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=20).read().decode())
    except urllib.error.HTTPError as e:
        return None

ouid = get('id', nickname='09이건호')['ouid']
ids = get('user/match', ouid=ouid, matchtype=60, offset=0, limit=12)

def dump(mid, label):
    md = get('match-detail', matchid=mid)
    if not md or len(md.get('matchInfo', [])) < 2:
        print(f'  {label}: 데이터 없음/오류'); return
    me = next((p for p in md['matchInfo'] if p['nickname'] == '09이건호'), None)
    opp = next((p for p in md['matchInfo'] if p['nickname'] != '09이건호'), None)
    if not me:
        print(f'  {label}: 내 데이터 없음'); return
    kst = (datetime.fromisoformat(md['matchDate']) + timedelta(hours=9)).strftime('%m-%d %H:%M')
    print(f"\n=== {label} | {kst} | {me['shoot'].get('goalTotal')}:{opp['shoot'].get('goalTotal')} vs {opp['nickname'][:14]} ===")
    rows = []
    for pl in me.get('player', []):
        st = pl['status']
        if st.get('spRating', 0) == 0:
            continue
        rows.append((pl['spPosition'], pl['spId'], st))
    for pos, spid, st in sorted(rows, key=lambda x: -x[0]):
        print(f"  pos={pos:>3} spId={spid:>10} 평점{st.get('spRating'):.1f}  "
              f"골{st.get('goal')} 도움{st.get('assist')} 슛{st.get('shoot')} 유효{st.get('effectiveShoot')} "
              f"패스{st.get('passSuccess')}/{st.get('passTry')} 드리블{st.get('dribbleSuccess')}/{st.get('dribbleTry')}")

# 스샷2 = 김재현롯데먹튀 1:0, 스샷1 = 바른구단주명9306 1:1
for mid in ids:
    md = get('match-detail', matchid=mid)
    if not md: continue
    opp = next((p for p in md['matchInfo'] if p['nickname'] != '09이건호'), None)
    if opp and '김재현' in opp['nickname']:
        dump(mid, '스샷2 김재현(1:0)')
    if opp and '바른구단' in opp['nickname']:
        dump(mid, '스샷1 바른구단(1:1)')

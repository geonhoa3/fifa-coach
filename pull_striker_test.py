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

SUA, FOR = 845176580, 845049072      # 수아레스, 포를란 spId
LM, RM = 845246191, 851194765        # 알바레스(좌2선), 그리즈만(우2선)
POS = {26: '좌', 20: '우'}

ouid = get('id', nickname='09이건호')['ouid']
ids = get('user/match', ouid=ouid, matchtype=60, offset=0, limit=40)

rows = []
for mid in ids:
    md = get('match-detail', matchid=mid)
    if not md or len(md.get('matchInfo', [])) < 2: continue
    me = next((p for p in md['matchInfo'] if p['nickname'] == '09이건호'), None)
    opp = next((p for p in md['matchInfo'] if p['nickname'] != '09이건호'), None)
    if not me: continue
    date = datetime.fromisoformat(md['matchDate'])
    if date.strftime('%m-%d') != '08-22': continue        # 오늘 판만
    ga = opp['shoot'].get('goalTotal')
    if ga is None or me['shoot'].get('goalTotal') is None:  # 몰수 제외
        continue
    pl = {p['spId']: p for p in me.get('player', [])}
    if SUA not in pl or FOR not in pl: continue
    s, f = pl[SUA], pl[FOR]
    lm, rm = pl.get(LM, {}).get('status', {}), pl.get(RM, {}).get('status', {})
    def st(p): return p['status']
    rows.append({
        'kst': (date + timedelta(hours=9)).strftime('%H:%M'),
        'score': f"{me['shoot']['goalTotal']}:{ga}",
        'opp': opp['nickname'][:12],
        'sua_pos': POS.get(s['spPosition'], '?'), 'sua': st(s),
        'for_pos': POS.get(f['spPosition'], '?'), 'for': st(f),
        'lm': lm, 'rm': rm,
    })

rows.sort(key=lambda r: r['kst'])
print(f"오늘(08-22) 스트라이커 테스트 판: {len(rows)}\n")
hdr = f"{'시각':>5} {'스코어':>6} {'상대':>12} | {'수아':>2}{'':1} {'평점':>4} {'골':>1} {'슛/유효':>6} | {'포를':>2}{'':1} {'평점':>4} {'골':>1} {'슛/유효':>6} | 좌2선도움/패스  우2선도움/패스"
print(hdr)
def sot(x): return f"{x.get('effectiveShoot',0)}/{x.get('shoot',0)}"
for r in rows:
    lm, rm = r['lm'], r['rm']
    print(f"{r['kst']:>5} {r['score']:>6} {r['opp']:>12} | "
          f"수아 {r['sua_pos']} {r['sua']['spRating']:>4.1f} {r['sua']['goal']} {sot(r['sua']):>6} | "
          f"포를 {r['for_pos']} {r['for']['spRating']:>4.1f} {r['for']['goal']} {sot(r['for']):>6} | "
          f"도움{lm.get('assist',0)} 패스{lm.get('passSuccess',0)}/{lm.get('passTry',0):<2}  "
          f"도움{rm.get('assist',0)} 패스{rm.get('passSuccess',0)}/{rm.get('passTry',0)}")

# 좌/우 자리별 집계 (누가 서든)
L = [(r['sua'] if r['sua_pos']=='좌' else r['for']) for r in rows]
R = [(r['sua'] if r['sua_pos']=='우' else r['for']) for r in rows]
def avg(lst,k): return sum(x[k] for x in lst)/len(lst) if lst else 0
print(f"\n[좌 스트라이커 자리] 평균평점 {avg(L,'spRating'):.2f}  총골 {sum(x['goal'] for x in L)}  평균슛 {avg(L,'shoot'):.1f}")
print(f"[우 스트라이커 자리] 평균평점 {avg(R,'spRating'):.2f}  총골 {sum(x['goal'] for x in R)}  평균슛 {avg(R,'shoot'):.1f}")
lwin = sum(1 for r in rows if (r['sua'] if r['sua_pos']=='좌' else r['for'])['spRating'] > (r['sua'] if r['sua_pos']=='우' else r['for'])['spRating'])
print(f"좌 > 우 평점: {lwin}/{len(rows)}판")
# 2선 관여
print(f"\n[좌2선 알바레스] 총도움 {sum(r['lm'].get('assist',0) for r in rows)}  평균패스 {sum(r['lm'].get('passTry',0) for r in rows)/len(rows):.1f}")
print(f"[우2선 그리즈만] 총도움 {sum(r['rm'].get('assist',0) for r in rows)}  평균패스 {sum(r['rm'].get('passTry',0) for r in rows)/len(rows):.1f}")

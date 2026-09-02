# -*- coding: utf-8 -*-
"""개인 대시보드 — 09이건호. 슛 히트맵 + 구역전환 + 시간대 + 슛수 승률."""
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
NICK = '09이건호'

# ---- 데이터 로드 ----
shots, games = [], []
for f in Path('data/api_cache').glob('*.json'):
    if f.name.startswith('_'): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    mi = d.get('matchInfo', [])
    if len(mi) < 2 or not d.get('matchDate'): continue
    me = next((p for p in mi if p['nickname'] == NICK), None)
    opp = next((p for p in mi if p['nickname'] != NICK), None)
    if not me or not opp: continue
    for s in (me.get('shootDetail') or []):
        if s.get('x') is not None:
            shots.append((s['x'], s['y'], s.get('result') == 3))
    gf = me['shoot'].get('goalTotal'); ga = opp['shoot'].get('goalTotal')
    if gf is None or ga is None: continue
    t = datetime.fromisoformat(d['matchDate']) + timedelta(hours=9)
    games.append({'h': t.hour, 'res': '승' if gf > ga else ('패' if gf < ga else '무'),
                  'shoot': me['shoot'].get('shootTotal') or 0})

# 메타 구역전환
def zone_conv(shot_list):
    zones = [(0, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    out = []
    for lo, hi in zones:
        z = [s for s in shot_list if lo <= s[0] < hi]
        out.append(100*sum(1 for s in z if s[2])/len(z) if z else 0)
    return out
who = json.loads(Path('data/api_cache_elite/_who.json').read_text(encoding='utf-8'))
eshots = []
for f in Path('data/api_cache_elite').glob('*.json'):
    if f.name.startswith('_'): continue
    d = json.loads(f.read_text(encoding='utf-8'))
    for p in d.get('matchInfo', []):
        if p['nickname'] in who:
            for s in (p.get('shootDetail') or []):
                if s.get('x') is not None:
                    eshots.append((s['x'], s['y'], s.get('result') == 3))

fig = plt.figure(figsize=(13, 8.5), dpi=130)
fig.suptitle(f'FIFA 개인 대시보드 — {NICK}  (최근 경기)', fontsize=15, fontweight='bold')

# ==== (1) 슛 히트맵 (피치) ====
ax1 = fig.add_subplot(2, 2, 1)
miss = [(x, y) for x, y, g in shots if not g]; goal = [(x, y) for x, y, g in shots if g]
# 피치: x=거리(0.7~1.0, 오른쪽=골), y=좌우(0~0.8)
ax1.add_patch(Rectangle((0.70, 0.0), 0.30, 0.80, fill=True, color='#2e7d32', alpha=0.10))
ax1.add_patch(Rectangle((0.90, 0.18), 0.10, 0.44, fill=False, ec='#888', lw=1))   # 박스
ax1.plot([1.0, 1.0], [0.30, 0.50], color='#333', lw=4)                             # 골대
ax1.scatter([x for x, y in miss], [y for x, y in miss], s=14, c='#bbb', alpha=0.5, label=f'빗나감 {len(miss)}')
ax1.scatter([x for x, y in goal], [y for x, y in goal], s=34, c='#2e7d32', edgecolor='white', linewidth=0.5, label=f'골 {len(goal)}', zorder=5)
ax1.axvline(0.90, color='#c62828', ls='--', lw=1, alpha=0.7)
ax1.text(0.895, 0.02, '박스선', color='#c62828', fontsize=8, ha='right')
ax1.set_xlim(0.70, 1.01); ax1.set_ylim(-0.02, 0.82)
ax1.set_title('① 내 슛 히트맵 — 어디서 쏘고 어디서 넣나', fontsize=11)
ax1.set_xlabel('← 멀다        상대 골대        가깝다 →'); ax1.set_yticks([])
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)

# ==== (2) 구역별 전환율 나 vs 메타 ====
ax2 = fig.add_subplot(2, 2, 2)
labels = ['먼거리\n(박스밖)', '중거리', '박스\n언저리', '박스안\n근접']
me_c, meta_c = zone_conv(shots), zone_conv(eshots)
x = np.arange(4); w = 0.38
ax2.bar(x - w/2, me_c, w, color='#1b6ec2', label='나')
ax2.bar(x + w/2, meta_c, w, color='#f9a825', label='메타(상위)')
for i, (a, b) in enumerate(zip(me_c, meta_c)):
    ax2.text(i - w/2, a + 1, f'{a:.0f}', ha='center', fontsize=8)
    ax2.text(i + w/2, b + 1, f'{b:.0f}', ha='center', fontsize=8, color='#a67c00')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel('골 전환율 %'); ax2.set_title('② 구역별 전환율 — 나 vs 메타', fontsize=11)
ax2.legend(fontsize=8); ax2.spines[['top', 'right']].set_visible(False)

# ==== (3) 시간대별 승률 ====
ax3 = fig.add_subplot(2, 2, 3)
bands = [(0, 5, '새벽\n0-6시'), (6, 11, '아침\n6-12시'), (12, 17, '오후\n12-18시'), (18, 23, '밤\n18-24시')]
xs, ys, ns = [], [], []
for lo, hi, lab in bands:
    g = [x for x in games if lo <= x['h'] <= hi]
    if len(g) < 5: continue
    xs.append(lab); ys.append(100*sum(1 for x in g if x['res'] == '승')/len(g)); ns.append(len(g))
cols = ['#c62828' if y < 45 else ('#2e7d32' if y > 55 else '#f9a825') for y in ys]
b = ax3.bar(xs, ys, color=cols)
for rect, y, n in zip(b, ys, ns):
    ax3.text(rect.get_x()+rect.get_width()/2, y+1, f'{y:.0f}%', ha='center', fontsize=9)
ax3.axhline(50, color='#888', ls=':', lw=1)
ax3.set_ylabel('승률 %'); ax3.set_ylim(0, 90); ax3.set_title('③ 시간대별 승률 (KST)', fontsize=11)
ax3.spines[['top', 'right']].set_visible(False)

# ==== (4) 슛 수 → 승률 ====
ax4 = fig.add_subplot(2, 2, 4)
sbins = [(0, 1, '0-1'), (2, 3, '2-3'), (4, 5, '4-5'), (6, 7, '6-7'), (8, 99, '8+')]
xs2, ys2, ns2 = [], [], []
for lo, hi, lab in sbins:
    g = [x for x in games if lo <= x['shoot'] <= hi]
    if not g: continue
    xs2.append(lab); ys2.append(100*sum(1 for x in g if x['res'] == '승')/len(g)); ns2.append(len(g))
ax4.plot(xs2, ys2, 'o-', color='#1b6ec2', lw=2, ms=8)
for xx, yy, n in zip(xs2, ys2, ns2):
    ax4.text(xx, yy+3, f'{yy:.0f}%', ha='center', fontsize=8.5, fontweight='bold')
ax4.axhline(50, color='#888', ls=':', lw=1)
ax4.set_ylabel('승률 %'); ax4.set_ylim(0, 95); ax4.set_xlabel('한 경기 슛 수')
ax4.set_title('④ 슛 수 → 승률 (경기 중 체크포인트)', fontsize=11)
ax4.spines[['top', 'right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
Path('reports/assets').mkdir(parents=True, exist_ok=True)
plt.savefig('reports/personal_dashboard.png', bbox_inches='tight')
print('saved reports/personal_dashboard.png')
print(f'슛 {len(shots)}개(골 {sum(1 for s in shots if s[2])}) · 경기 {len(games)}판')

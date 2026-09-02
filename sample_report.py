# -*- coding: utf-8 -*-
"""하위 유저 샘플 메타 갭 리포트 — 슛 위치까지 파고든 '왜'."""
import json
from pathlib import Path

import sys
TARGET = sys.argv[1] if len(sys.argv) > 1 else 'TARGET_NICKNAME'  # 분석 대상 유저 닉네임 (인자로 전달)

def shots_of(nick, folders):
    out = []
    for folder in folders:
        for f in Path(folder).glob('*.json'):
            if f.name.startswith('_'): continue
            d = json.loads(f.read_text(encoding='utf-8'))
            for p in d.get('matchInfo', []):
                if p['nickname'] == nick:
                    for s in (p.get('shootDetail') or []):
                        if s.get('x') is not None:
                            out.append(s)
    return out

def elite_shots():
    who = json.loads(Path('data/api_cache_elite/_who.json').read_text(encoding='utf-8'))
    out = []
    for f in Path('data/api_cache_elite').glob('*.json'):
        if f.name.startswith('_'): continue
        d = json.loads(f.read_text(encoding='utf-8'))
        for p in d.get('matchInfo', []):
            if p['nickname'] in who:
                for s in (p.get('shootDetail') or []):
                    if s.get('x') is not None:
                        out.append(s)
    return out

ZONES = [(0.0, 0.85, '먼 거리(박스 밖)'), (0.85, 0.90, '중거리'),
         (0.90, 0.95, '박스 언저리'), (0.95, 1.01, '박스 안 근접')]

def breakdown(shots):
    n = len(shots); g = sum(1 for s in shots if s.get('result') == 3)
    rows = []
    for lo, hi, lab in ZONES:
        z = [s for s in shots if lo <= s['x'] < hi]
        zg = sum(1 for s in z if s.get('result') == 3)
        rows.append((lab, len(z), 100*len(z)/n if n else 0, 100*zg/len(z) if z else 0))
    return n, g, 100*g/n if n else 0, rows

t_shots = shots_of(TARGET, ['data/api_cache_lowtier'])
e_shots = elite_shots()
tn, tg, tconv, trows = breakdown(t_shots)
en, eg, econv, erows = breakdown(e_shots)

print(f"===== 메타 갭 리포트: {TARGET} =====\n")
print(f"총 슛 {tn}개 · 골 {tg}개 · 전환율 {tconv:.1f}%   (메타 상위권 {econv:.1f}%)")
print(f"→ 슛 하나당 {econv-tconv:.1f}%p 손해\n")

print(f"{'구역':>14} | {'내 슛비중':>7} {'내 전환':>6} | {'메타 슛비중':>8} {'메타 전환':>7}")
for (tl, tc, tp, tcv), (el, ec, ep, ecv) in zip(trows, erows):
    flag = '  ← 여기 샘' if (tp - ep > 5 and tcv < 25) else ''
    print(f"{tl:>14} | {tp:>6.0f}% {tcv:>5.0f}% | {ep:>7.0f}% {ecv:>6.0f}%{flag}")

# 반사실: 슛 분포를 메타처럼 바꾸면 골 몇 개 더?
# 내 슛 수를 메타 구역비중으로 재배분 × 메타 구역전환율
exp_goals = 0
for (lo, hi, lab), (el, ec, ep, ecv) in zip(ZONES, erows):
    exp_goals += tn * (ep/100) * (ecv/100)
print(f"\n[반사실] 같은 슛 수({tn}개)를 메타처럼 쐈다면 예상 골: {exp_goals:.0f}개 (실제 {tg}개)")
print(f"→ 잠재 손실 약 {exp_goals-tg:.0f}골")

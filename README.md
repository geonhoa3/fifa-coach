# FIFA Coach — FC 온라인 영상 자동 분석 코칭 시스템

FC 온라인(FIFA Online 4) 게임 영상을 입력으로 받아 **미니맵 기반 자동 분석 + xT 가치 평가 + 코칭 리포트**를 생성하는 시스템.

---

## 한 줄 요약

> 영상 한 개 넣으면 → "이 시점 행동 +0.86 (좋음)", "저 시점 행동 -0.50 (백패스, 기회 놓침)" 같은 자동 코칭 리포트 PNG 한 장 출력.

---

## 핵심 아이디어 (정공법 vs 판 뒤집기)

| 정공법 (불가능) | 우리 방식 (가능) |
|---|---|
| 3D 중계 화면 통째 → 선수/공 인식 → tactical AI | **미니맵 (2D 추상화) 만 사용** |
| ML 모델 학습 필요 | OpenCV + 통계만 사용 |
| 데이터 라벨링만 1년 | 영상 1개로도 작동 |

FC 온라인 미니맵에는 게임 상태가 친절하게 abstracted 되어 있음. 그걸 직접 읽음.

---

## 데이터 흐름

```
영상 (mp4)
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ extract_game_state                                  │
│   미니맵 → 공/조작선수/양팀 dot 매 프레임 추출          │
│   → game_state.csv                                  │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ analyze_motion                                      │
│   공 속도 → hold / dribble / pass_or_shot 분류         │
│   → motion.csv                                      │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ detect_events                                       │
│   빨강 dot 검출률로 빨강 유니폼 케이스 자동 감지         │
│   pass_or_shot → shot / pass_success / intercepted   │
│                / opponent_pass                       │
│   → events.csv                                      │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ grid                                                │
│   미니맵 → 16x12 그리드. 공격 방향 정규화              │
│   → events.csv 에 ball_i_norm, ball_j_norm 컬럼 추가  │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ xt_model                                            │
│   각 그리드 셀 "N초 안에 골대 영역 도달" 비율 = V(i,j)  │
│   → xt_values.json, xt_heatmap.png                  │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ evaluate_actions                                    │
│   매 패스/슛 의 xt_change = V(도착) - V(출발)         │
│   → events.csv 에 xt_change 추가                     │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ coaching_report                                     │
│   |xt_change| > 0.3 인 행동만 → 미니맵 카드 격자       │
│   → coaching_report.png                             │
└─────────────────────────────────────────────────────┘
```

---

## 사용법

### 0. 환경 세팅 (1회)

```bash
pip install -r requirements.txt
```

### 1. 미니맵 영역 잡기 (영상 종류당 1회)

```bash
python -m src.minimap calibrate data/videos/match01.mp4
```

마우스로 미니맵 영역 드래그 → ENTER. `data/minimap_roi.json` 에 저장됨.

### 2. 영상 처리 (한 영상씩 또는 일괄)

**한 영상:**
```bash
python -m src.process_all data/videos/match01.mp4
```

**여러 영상 일괄 (강추):**
```bash
python -m src.process_all data/videos/match*.mp4
```

각 영상마다 약 7~8분. 결과는 `data/output/<영상명>/` 폴더에 자동 저장.

### 3. V 통합 (영상 여러 개 모인 뒤)

```bash
python -m src.combine_xt match04 match05 match06 match07 match08
```

5개 이상 영상이면 V_user 안정화. `data/output/combined/V_user.json` 생성.

### 4. V_user 기반 재평가

```bash
python -m src.evaluate_all_with_user
```

영상별 events.csv 의 xt_change 를 통합 V_user 기준으로 재계산.

### 5. 코칭 리포트 생성

```bash
python -m src.coaching_report_all
```

각 영상 폴더에 `coaching_report.png` 생성.

### 6. (선택) 통계지 OCR 비교

영상에 통계지 포함된 경우:
```bash
python -m src.parse_stats data/test/stats_match02.png
python -m src.compare
```

정답 vs 우리 시스템 결과 비교 → 정확도 리포트.

---

## 결과물 위치

| 파일 | 의미 |
|---|---|
| `data/output/<영상>/game_state.csv` | 매 프레임 공/선수 위치 |
| `data/output/<영상>/motion.csv` | 공 속도 + 행동 분류 |
| `data/output/<영상>/events.csv` | 행동 + outcome + xT |
| `data/output/<영상>/xt_values.json` | 영상별 V(i,j) |
| `data/output/<영상>/xt_heatmap.png` | 영상별 가치맵 |
| `data/output/<영상>/coaching_report.png` | **자동 코칭 리포트 (메인 출력)** |
| `data/output/combined/V_user.json` | 통합 V (사용자 평균) |
| `data/output/combined/V_user_heatmap.png` | 통합 가치맵 |
| `data/output/combined/V_compare.png` | V_user vs V_pro 비교 (V_pro 있을 때) |

---

## 알고리즘 핵심

### 빨강 유니폼 자동 감지

| 우리팀 유니폼 | 빨강 dot 검출률 | is_our_possession 로직 |
|---|---|---|
| 흰/남색 등 (정상) | ~30% (우리 공 안 가질 때만 빨강) | 빨강 안 보임 = 우리 소유 |
| 빨강 | ~75% (조작 시 dot이 빨강) | 빨강 보임 = 우리 소유 (반전) |

`detect_events.detect_red_uniform_case()` 가 50% 임계값으로 자동 분기.

### xT 가치 (단순화 버전)

```
V(i, j) = (i, j) 셀에서 시작한 점유가 N초 안에
          상대 골대 영역(i_norm >= 12) 도달한 비율
```

- 영상 1개로도 작동
- 영상 누적될수록 정밀해짐 (combine_xt 로 통합)
- 슛 직접 학습보다 신호 풍부 (한 점유에서 골대 도달이 슛보다 자주 발생)

### 행동 가치

```
xt_change = V(도착) - V(출발)
```

- 양수: 좋은 행동 (가치 증가)
- 음수: 나쁜 행동 (가치 감소, 백패스 등)
- shot: 도착이 골대 영역이라 V_max=1.0 강제

---

## 한계 (PoC 단계)

| 한계 | 영향 | 향후 |
|---|---|---|
| 선수 식별 안 함 | 누가 누구에게 패스했는지 모름 | 등번호 OCR / 얼굴 인식 추가 |
| 짧은 안전 패스 안 잡힘 | 패스 성공률 통계지 92% vs 우리 시스템 64% | 임계값 더 낮추기 |
| 미니맵 안 보이는 시점 (셀럽 등) | 약 25~30% 데이터 누락 | 메인 화면 fallback 분석 |
| 영상별 미니맵 위치 다르면 깨짐 | match03 같은 다른 해상도 영상 안 됨 | 영상별 ROI 자동 검출 |
| V 통계가 한 영상으로는 노이즈 | V=1.00 같은 단발 셀 | 영상 5개+ 누적 |

---

## 다음 단계 (개발 로드맵)

1. **고수 영상 V_pro 학습** — `combine_xt --output V_pro`
2. **V_user vs V_pro 비교 코칭** — `compare_xt.py` (이미 준비됨)
3. **메인 화면 OCR** — 코너킥, 골 자동 검출
4. **선수 식별** — 등번호 / 얼굴 인식
5. **자동 영상 편집** — 하이라이트 클립 추출 (xT 큰 변화 시점 기준)
6. **다양한 해상도 대응** — 영상별 ROI 자동 검출

---

## 검증 결과 (match02 기준)

| 지표 | 정답 (통계지) | 우리 시스템 | 오차 |
|---|---|---|---|
| 슛 (우리) | 8 | 7 | -1 (12%) |
| 패스 성공률 | 92% | 64% | -28%p (위험 패스만 분석함) |
| 점유율 | 53% | 43% | -10%p (받아들일 수준) |

슛 카운트 정확, 점유율 합리적. 패스 성공률은 측정 단위 차이 (시스템은 큰 점프 = 위험 패스만 잡음).

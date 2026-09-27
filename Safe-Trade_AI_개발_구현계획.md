# Safe-Trade AI 개발 구현 계획

## 1. 개발 목표

본 개발은 **TRON Mainnet의 USDT(TRC-20) 거래 데이터**를 기반으로 지갑 간 자금 이동의 시간적·구조적 행동 패턴을 분석하고, AML 위험지갑을 우선 식별하는 **Python 기반 Risk Scoring PoC**를 구축하는 것을 목표로 한다.

핵심 흐름은 다음과 같다.

```text
TRON Mainnet
     ↓
TronGrid API
     ↓
USDT(TRC-20) 거래 데이터 수집
     ↓
지갑별 Behavioral Feature 생성
     ↓
Positive / Negative Proxy 구성
     ↓
ML 모델 학습 및 비교
     ↓
신규 지갑 Positive Class Probability 산출
     ↓
Risk Score(0~100)
     ↓
Feature Importance / 주요 위험 Feature 제시
     ↓
조사 우선순위화
```

본 시스템의 Risk Score는 보이스피싱 또는 자금세탁 여부를 확정하는 값이 아니라 **AML 위험 신호 및 추가 분석 우선순위를 나타내는 내부 지표**로 사용한다.

---

# 2. 전체 시스템 구조

## 2.1 3-Layer 구조

### 1-Layer — On-chain Data Collection

실제 TRON Mainnet에서 USDT(TRC-20) 거래 데이터를 수집한다.

수집 대상 주요 필드:

- `from`: 송신 지갑
- `to`: 수신 지갑
- `amount`: USDT 거래금액
- `timestamp`: 거래 시각
- `transaction_id`: 거래 ID

### 2-Layer — Behavioral Feature + ML

온체인 거래 데이터를 지갑 단위 Feature로 변환한다.

주요 Feature:

- Fan-in
- Fan-out
- Smurfing
- Holding Time
- Pass-through Ratio
- Transaction Count
- 거래금액 통계

이후 Positive Proxy와 Negative Proxy의 거래행동 차이를 ML 모델이 학습한다.

비교 모델:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### 3-Layer — Risk Scoring / 조사 우선순위화

학습된 모델을 신규·미라벨 지갑에 적용한다.

```text
신규 지갑
   ↓
Feature 계산
   ↓
ML 예측
   ↓
Positive Class Probability
   ↓
Risk Score 0~100
   ↓
주요 위험 Feature
   ↓
추가 분석 우선 대상
```

---

# 3. 개발 폴더 구조

```text
safe_trade_ai/
│
├── data/
│   ├── raw/
│   │   └── tron_usdt_transactions.csv
│   │
│   ├── processed/
│   │   └── wallet_features.csv
│   │
│   └── labels/
│       └── proxy_labels.csv
│
├── src/
│   ├── config.py
│   ├── trongrid_collector.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── label_builder.py
│   ├── train.py
│   ├── evaluate.py
│   ├── risk_scoring.py
│   └── explain.py
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── risk_scores.csv
│   └── feature_importance.csv
│
└── main.py
```

---

# 4. 1단계 — TRON USDT 데이터 수집

## 4.1 데이터 수집 방식

TronGrid API를 활용하여 TRON Mainnet의 USDT(TRC-20) Transfer 데이터를 수집한다.

처음부터 TRON 전체 거래를 수집하는 방식보다는 **Seed → 주변 지갑 확장 방식**으로 PoC를 구축한다.

```text
위험 Proxy 주소
      ↓
해당 주소 거래 조회
      ↓
거래 상대방 주소 추출
      ↓
상대방 거래 조회
      ↓
관련 지갑의 행동 Feature 생성
```

## 4.2 수집 데이터

최소한 다음 구조를 확보한다.

| 필드 | 설명 |
|---|---|
| from | 송신 지갑 |
| to | 수신 지갑 |
| amount | USDT 거래금액 |
| timestamp | 거래 시각 |
| transaction_id | 거래 ID |

수집 결과 예시:

```text
data/raw/tron_usdt_transactions.csv
```

---

# 5. 2단계 — 데이터 전처리

수집된 원본 데이터를 분석 가능한 형태로 변환한다.

주요 처리:

- timestamp → datetime 변환
- USDT raw value → 실제 USDT 금액 변환
- 중복 transaction 제거
- `from`, `to` 주소 정규화
- 잘못된 거래값 제거
- 거래시간 기준 정렬

예상 데이터 구조:

```text
transaction_id
timestamp
from
to
amount
```

---

# 6. 3단계 — 지갑별 Behavioral Feature 생성

## 6.1 Fan-in

여러 지갑에서 특정 지갑으로 자금이 집금되는 정도를 나타낸다.

```text
A ──┐
B ──┤
C ──┼──> X
D ──┤
E ──┘
```

기본 Feature:

```python
fan_in = 해당 지갑으로 송금한 고유 지갑 수
```

---

## 6.2 Fan-out

특정 지갑에서 여러 지갑으로 자금이 분산되는 정도를 나타낸다.

```text
             ┌──> B
             ├──> C
A ───────────┼──> D
             ├──> E
             └──> F
```

기본 Feature:

```python
fan_out = 해당 지갑이 송금한 고유 수취 지갑 수
```

---

## 6.3 Smurfing

자금을 여러 거래로 분할·분산하는 행동을 정량화한다.

예:

```text
10,000 USDT
     ↓
2,000
2,100
1,900
2,000
2,000
```

초기 PoC에서는 다음과 같은 거래금액 통계 Feature를 활용한다.

- transaction_count
- amount_mean
- amount_median
- amount_std
- amount_cv
- small_tx_ratio

예:

```text
CV = 표준편차 / 평균
```

---

# 7. 4단계 — Holding Time 계산

Holding Time은 자금 수취 후 다음 출금까지 걸리는 시간을 계산한다.

예:

```text
09:00
A → X
10,000 USDT

09:07
X → B
9,900 USDT
```

```text
Holding Time = 7분
```

지갑별로 다음 Feature를 생성한다.

- holding_time_mean
- holding_time_median
- holding_time_min

특히 **입금→출금 시간의 중앙값**을 주요 Feature로 활용한다.

---

# 8. 5단계 — Pass-through Ratio 계산

받은 자금이 일정 시간 안에 다시 외부로 이동하는 비율을 계산한다.

기본 정의:

```text
Pass-through Ratio
=
일정 시간 내 재출금 금액
/
입금 금액
```

예:

```text
입금 = 10,000 USDT
재출금 = 9,800 USDT

Pass-through Ratio = 0.98
```

지갑별로 일정 시간창을 설정하여 계산한다.

---

# 9. 6단계 — ML 학습 데이터 구성

## 9.1 Positive / Negative Proxy

본 PoC에서는 실제 보이스피싱 지갑 Label을 직접 사용하는 것이 아니라 공개적으로 식별 가능한 주소를 Proxy로 활용한다.

예:

| 구분 | Label | ML 의미 |
|---|---:|---|
| OFAC SDN TRON 주소 | 1 | Positive Risk Proxy |
| 검증된 거래소·발행사 주소 | 0 | Negative Proxy |
| 일반 신규 지갑 | ? | 실제 탐지 대상 |
| 경찰청 보이스피싱 통계 | - | 문제정의·배경자료 |
| 관세청 환전상 단속정보 | - | 문제정의·배경자료 |

중요:

```text
경찰청 통계
관세청 단속정보
        ↓
ML Label로 사용하지 않음
```

이 자료들은 문제정의와 배경자료로 활용한다.

---

# 10. 7단계 — ML 모델 비교

동일한 Feature와 동일한 학습/검증 분할을 사용하여 다음 모델을 비교한다.

```python
Logistic Regression
Random Forest
Gradient Boosting
```

평가지표:

- Precision
- Recall
- F1-score
- PR-AUC

결과 예시:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |

실제 값은 학습 데이터와 검증 결과를 실행한 후 기록한다.

---

# 11. 8단계 — 신규 지갑 위험도 산출

학습 완료 후 Label이 없는 일반 신규 지갑을 모델에 입력한다.

```text
신규 지갑
     ↓
Behavioral Feature
     ↓
학습된 ML 모델
     ↓
Positive Class Probability
```

예:

```text
Positive Probability = 0.87
```

이를 내부 Risk Score로 변환한다.

```python
risk_score = probability * 100
```

예:

```text
0.87 → 87
```

단, 이 값은 **범죄 확률의 확정적 의미가 아니라 모델 내부 위험 신호 점수**로 설명한다.

---

# 12. Risk Score 결과

최종 결과는 다음과 같이 구성한다.

```text
Wallet
TXXXXXXXXXXXXXXXXXXXXXXXX

Risk Score
87 / 100

Positive Probability
0.87

Behavior Features
-------------------------
Fan-in                34
Fan-out               51
Transaction Count    127
Holding Time          18 min
Pass-through Ratio   0.94
Smurfing Index        0.72
```

그리고 조사 우선순위를 표시한다.

```text
[추가 분석 우선 대상]
```

단, 다음 문구를 함께 표시한다.

```text
※ 본 Risk Score는 보이스피싱 또는 자금세탁 여부를 확정하지 않음.
※ 온체인 거래행동 기반 AML 위험 신호 및 추가 분석 우선순위를 나타내는 내부 지표임.
```

---

# 13. Feature Importance / 설명 기능

ML 결과와 함께 주요 위험 Feature를 제시한다.

예:

```text
주요 위험 Feature
-------------------------

1. Pass-through Ratio
2. Fan-out
3. Holding Time
4. Transaction Count
5. Fan-in
```

모델에 따라 Feature Importance 또는 SHAP 등을 활용한다.

목적은 단순히

```text
Risk Score = 87
```

을 보여주는 것이 아니라,

```text
왜 위험 신호가 높게 산출되었는가?
```

를 설명하는 것이다.

---

# 14. 최종 PoC 처리 흐름

```text
┌─────────────────────────────┐
│       TRON Mainnet          │
│       USDT(TRC-20)          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       TronGrid API           │
│       On-chain Collection    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       Data Preprocessing     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│    Behavioral Features      │
│ Fan-in / Fan-out / Smurfing │
│ Holding / Pass-through      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│          ML Training         │
│ LR / RF / Gradient Boosting │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Model Evaluation        │
│ Precision / Recall / F1     │
│ PR-AUC                       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       New Wallet Input       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Positive Class Probability   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Risk Score 0~100        │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Feature Importance / SHAP    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   조사 우선순위 지갑 목록    │
└─────────────────────────────┘
```

---

# 15. 개발 시 반드시 구분할 데이터

## ML 학습 데이터

```text
Positive Proxy
Negative Proxy
       ↓
Behavioral Feature
       ↓
ML 학습
```

## 실제 탐지 대상

```text
일반 신규 지갑
Label = ?
       ↓
학습된 모델
       ↓
Risk Score
```

## 배경자료

```text
경찰청 보이스피싱 통계
관세청 환전상 단속정보
       ↓
문제정의 및 개발 필요성 설명
```

세 데이터의 역할을 혼합하지 않는다.

---

# 16. 개발 단계별 목표

## Phase 1 — 데이터 수집

목표:

```text
실제 TRON USDT 거래 CSV 생성
```

완료 조건:

- 실제 Mainnet 데이터 확보
- from/to/amount/timestamp/transaction_id 확보
- 중복 및 오류 데이터 처리

## Phase 2 — Feature Engineering

목표:

```text
wallet_features.csv
```

생성.

포함 Feature:

```text
fan_in
fan_out
transaction_count
amount_mean
amount_median
amount_std
amount_cv
holding_time_mean
holding_time_median
pass_through_ratio
```

## Phase 3 — Proxy Label

목표:

```text
proxy_labels.csv
```

생성.

```text
Positive = 1
Negative = 0
```

## Phase 4 — ML

3개 모델 비교:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

평가:

```text
Precision
Recall
F1
PR-AUC
```

## Phase 5 — Risk Scoring

```text
Probability
      ↓
Risk Score 0~100
```

## Phase 6 — 설명 및 결과 출력

```text
Risk Score
+
주요 위험 Feature
+
조사 우선순위
```

---

# 17. 최종 산출물

개발 완료 후 다음 결과물을 확보한다.

```text
① 실제 TRON USDT 거래 데이터
② 지갑별 Feature Dataset
③ Proxy Label Dataset
④ ML 모델별 성능 비교 결과
⑤ 최종 모델
⑥ 신규 지갑 Risk Score
⑦ Feature Importance
⑧ 조사 우선순위 결과
⑨ 실행 가능한 Python 코드
⑩ 개발 결과 및 한계점
```

---

# 18. 핵심 개발 원칙

1. 실제 TRON Mainnet USDT 데이터를 우선 사용한다.
2. 경찰청 보이스피싱 통계는 ML Label로 사용하지 않는다.
3. 관세청 환전상 단속정보 역시 ML Feature/Label로 사용하지 않는다.
4. Positive Label은 실제 보이스피싱 확정 지갑이 아닌 **위험주소 Proxy**임을 명시한다.
5. 신규 지갑은 학습 대상이 아니라 **실제 탐지 대상**으로 구분한다.
6. Risk Score는 범죄 여부를 확정하지 않는다.
7. F1과 PR-AUC를 중심으로 모델을 평가한다.
8. Risk Score와 함께 주요 위험 Feature를 제시한다.
9. PoC 결과를 상용 서비스 수준의 탐지 성능으로 과장하지 않는다.
10. 최종 목적은 범죄 판정이 아니라 **AML 위험 신호 탐지 및 조사 우선순위화 가능성 검증**이다.

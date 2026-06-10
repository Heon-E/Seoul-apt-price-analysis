# 서울 아파트 가격 결정요인 분석

기준금리, 주담대금리, 소비자심리지수, 주택 공급, 정권 효과를 중심으로 서울 아파트 가격 변동률을 계량 분석한 개인 경제 리서치 프로젝트입니다.

- 분석 기간: 2015년 1월 ~ 2025년 10월 (130개월)
- 작성자: 구자헌

---

## 분석 단계 및 주요 결과

### 1단계 — 단순회귀 (기준금리 → 집값 변동률)
- R² = 0.088, p = 0.001
- 기준금리 1%p 상승 시 집값 변동률 −0.19%p
- **한계**: 기준금리 하나로는 집값의 8.8%밖에 설명 못 함

### 2단계 — 다중회귀 (6개 변수)
- R² = 0.456 (설명력 대폭 향상)
- 유의 변수: 주담대금리(−0.722), CSI(+0.047), 윤석열더미(+0.878)
- **한계**: Durbin-Watson = 0.700 → 강한 자기상관 잔존

### 3단계 — 시차 효과 분석 (주택인허가)
- 6 / 12 / 24개월 시차 적용
- 24개월 시차에서 이론 부합 방향이나 모든 시차에서 통계적 비유의
- **결론**: 인허가 데이터로는 공급 효과 검증 불가 → 준공 건수 재분석 필요

### 4단계 — Newey-West HAC 보정
- 자기상관 존재 시 표준오차 조정
- 핵심 변수 유의성 유지 (주담대금리 p=0.002, CSI p=0.000)
- **한계**: 자기상관을 보정만 했을 뿐 구조적으로 해결하지 못함

### 5단계 — ARIMAX(1,0,0)
- AR(1) 항 추가로 자기상관 구조적 해결
- **DW: 0.292 → 1.662** (자기상관 대부분 해소)
- 강건 변수: 주담대금리(p=0.017), CSI(p=0.031), ar.L1 φ=0.765(p=0.000)
- **핵심 발견**: HAC에서 유의했던 윤석열더미(p=0.012)가 ARIMAX에서 소멸(p=0.457) → 관성을 통제하면 정치 효과는 설명되지 않음

---

## 프로젝트 구조

```
seoul house pricing/
├── data/               원본 및 병합 데이터
│   ├── merged_multi.csv    분석용 통합 데이터
│   ├── base_rate.csv
│   ├── mortgage_rate.csv
│   ├── csi.csv
│   └── house_permit.csv
├── scripts/            분석 스크립트
│   ├── collect_data.py         데이터 수집 및 병합
│   ├── regression_multi.py     다중회귀 + VIF
│   ├── regression_lag.py       시차 분석
│   ├── regression_hac.py       HAC 보정
│   ├── arimax_acf.py           ACF/PACF 진단
│   └── arimax_model.py         ARIMAX(1,0,0) 분석
├── figures/            시각화 결과
├── report/             분석 보고서
│   ├── seoul_apt_report_v2.md  1~4단계 종합 보고서
│   └── arimax_analysis.md      ARIMAX 분석 보고서
└── obsidian_notes/     학습 노트
```

## 실행 순서

```bash
python scripts/collect_data.py    # 데이터 병합
python scripts/regression_multi.py
python scripts/regression_lag.py
python scripts/regression_hac.py
python scripts/arimax_acf.py      # ACF/PACF 진단
python scripts/arimax_model.py    # ARIMAX 분석
```

필요 라이브러리: `pandas`, `matplotlib`, `statsmodels`

## 다음 과제

- 준공 건수 데이터로 공급 효과 재검증
- KDI 논문 리뷰 — 이론적 근거 보강

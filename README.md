# 기준금리와 서울 아파트 매매가격의 관계 분석

한국은행 기준금리와 서울 아파트 매매가격지수(2015.01~2025.10, 월별 130개월) 사이의 관계를 데이터로 검증해본 개인 경제 리서치 프로젝트입니다.

## 연구 질문

> 기준금리의 변동은 서울 아파트 가격 변동률에 통계적으로 유의한 영향을 미치는가?

## 데이터

| 변수명 | 설명 | 출처 |
|---|---|---|
| `base_rate` | 한국은행 기준금리 (%) | 한국은행 ECOS |
| `seoul_apt_price_index` | 서울 아파트 매매가격지수 | 한국부동산원 R-ONE |
| `mom_change_pct` | 전월 대비 가격 변동률 (%) | 한국부동산원 R-ONE |

분석 기간: 2015년 1월 ~ 2025년 10월 (130개월)

## 분석 방법

1. **데이터 병합**: 두 출처의 월별 데이터를 `date` 기준으로 병합 (`scripts/merge_data.py`)
2. **탐색적 분석(EDA)**: 기술통계, 시계열 그래프, 상관관계 분석
3. **회귀분석(OLS)**: 처음에는 가격지수 자체를 종속변수로 사용했으나 자기상관(Durbin-Watson ≈ 0.007)이 심해 이론과 반대 부호가 나옴 → 종속변수를 전월 대비 변동률로 바꿔 재분석 (`scripts/eda.py`)

## 주요 결과

- 기준금리 1%p 상승 시 서울 아파트 가격 월간 상승률이 평균 **0.19%p 감소** (계수 −0.1916, P = 0.001)
- R² = 0.088 → 기준금리는 집값 변동의 **8.8%만 설명**. 나머지는 공급, 유동성, 정책, 심리 등 다른 요인
- 2020~2022년 초저금리 구간은 이론적 관계에서 벗어난 이상 구간 (유동성 폭발 + 공급 부족)

자세한 해석과 한계는 [`report/seoul_apt_report.md`](report/seoul_apt_report.md)를 참고하세요.

## 프로젝트 구조

```
seoul house pricing/
├── data/            원본·병합 데이터 (base_rate.csv, seoul_apt_price_index.csv, merged.csv)
├── scripts/         분석 코드 (merge_data.py, eda.py)
├── figures/         시각화 결과 (timeseries.png, scatter.png)
├── report/          최종 보고서 (seoul_apt_report.md)
└── notes/           학습 메모 (개인용)
```

## 실행 방법

프로젝트 루트(`seoul house pricing/`)에서 순서대로 실행합니다.

```bash
python scripts/merge_data.py   # data/merged.csv 생성
python scripts/eda.py          # EDA + 회귀분석 + figures/ 그래프 생성
```

필요 라이브러리: `pandas`, `matplotlib`, `statsmodels`

## 한계와 다음 단계

- 단일 변수(금리)만으로는 설명력이 낮음 → 주택 공급량, 가계부채, 전세가율 등을 추가한 **다중회귀분석** 필요
- 변환 후에도 자기상관(Durbin-Watson 0.378)이 남아있음 → 금리 변화의 **시차효과(lag)**를 명시적으로 모델링하는 후속 분석 예정

## 작성자

구자헌 (경제학 전공, 통계학 복수전공 예정)

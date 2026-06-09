"""
ARIMAX 사전 분석: ACF / PACF 시각화
- 목적: AR 차수(p)와 MA 차수(q)를 결정하기 위한 진단
- 입력: data/merged_multi.csv
- 출력: figures/acf_pacf.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
import os

# ── 데이터 로드 ──────────────────────────────────────────────
DATA_PATH = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/data/merged_multi.csv"
)
OUT_PATH = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/figures/acf_pacf.png"
)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").dropna(subset=["mom_change_pct"])
y = df["mom_change_pct"]

# ── Durbin-Watson 재확인 ─────────────────────────────────────
dw = durbin_watson(y)
print(f"Durbin-Watson (종속변수 자체): {dw:.3f}")
print(f"관측치 수: {len(y)}")
print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ── ACF / PACF 그래프 ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("ACF / PACF — 서울 아파트 집값 변동률 (mom_change_pct)", fontsize=13, y=0.98)

plot_acf(y, lags=24, ax=axes[0], alpha=0.05)
axes[0].set_title("ACF (자기상관함수) — 막대가 점선 밖: 그 시차가 통계적으로 유의미함", fontsize=11)
axes[0].set_xlabel("시차 (개월)")
axes[0].set_ylabel("상관계수")

plot_pacf(y, lags=24, ax=axes[1], alpha=0.05, method="ywm")
axes[1].set_title("PACF (편자기상관함수) — AR 차수(p) 결정에 사용", fontsize=11)
axes[1].set_xlabel("시차 (개월)")
axes[1].set_ylabel("편상관계수")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\n저장 완료: {OUT_PATH}")
plt.show()

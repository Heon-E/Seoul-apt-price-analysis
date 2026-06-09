"""
ARIMAX(1,0,0) 분석
- 목적: DW=0.292 자기상관을 AR(1) 항으로 모형 내 처리
- 설명변수(X): mortgage_rate, csi, moon_dummy, yoon_dummy
- 입력: data/merged_multi.csv
- 출력: 콘솔 결과 + figures/arimax_result.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

# ── 데이터 로드 ──────────────────────────────────────────────
import os
DATA_PATH = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/data/merged_multi.csv"
)
OUT_PATH = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/figures/arimax_result.png"
)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").dropna(subset=["mom_change_pct", "mortgage_rate", "csi"])

# ── 변수 설정 ────────────────────────────────────────────────
y = df["mom_change_pct"]
X = df[["mortgage_rate", "csi", "moon_dummy", "yoon_dummy"]]

# ── ARIMAX(1,0,0) 모형 적합 ──────────────────────────────────
model = SARIMAX(
    endog=y,
    exog=X,
    order=(1, 0, 0),   # AR(1), 차분 없음, MA 없음
    trend="c"          # 상수항 포함
)
result = model.fit(disp=False)

print("=" * 60)
print("ARIMAX(1,0,0) 결과 요약")
print("=" * 60)
print(result.summary())

# ── 잔차 자기상관 확인 ───────────────────────────────────────
residuals = result.resid
dw = durbin_watson(residuals)
print(f"\n[진단] 잔차 Durbin-Watson: {dw:.3f}")
print(f"  → 2에 가까울수록 자기상관 없음 (이전: 0.292)")
if dw > 1.5:
    print("  → 자기상관 상당 부분 해소됨")
else:
    print("  → 아직 자기상관 잔존 — 차수 조정 필요")

# ── 시각화: 실제값 vs 예측값 + 잔차 ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 실제 vs 예측
fitted = result.fittedvalues
axes[0].plot(df["date"], y.values, label="Actual", linewidth=1.2, color="#1a1a2e")
axes[0].plot(df["date"], fitted.values, label="ARIMAX Fitted",
             linewidth=1.2, color="#e74c3c", linestyle="--")
axes[0].set_title("ARIMAX(1,0,0): Actual vs Fitted — Seoul Apt Price Change (%)")
axes[0].set_ylabel("MoM Change (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 잔차
axes[1].bar(df["date"], residuals.values, color="#2980b9", alpha=0.6, width=20)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title(f"Residuals (DW={dw:.3f}  |  target: ~2.0)")
axes[1].set_ylabel("Residual")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장: {OUT_PATH}")
plt.show()

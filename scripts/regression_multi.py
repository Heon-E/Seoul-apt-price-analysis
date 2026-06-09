"""
다중회귀분석 스크립트
=====================
종속변수  : mom_change_pct (서울 아파트 가격 전월대비 변동률)
독립변수  : base_rate, mortgage_rate, csi, housing_permit,
            moon_dummy, yoon_dummy (park_dummy = baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
for font in fm.fontManager.ttflist:
    if 'AppleGothic' in font.name or 'Malgun' in font.name:
        plt.rcParams['font.family'] = font.name
        break

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "..", "data")
fig_dir    = os.path.join(script_dir, "..", "figures")
os.makedirs(fig_dir, exist_ok=True)

# ── 1. 데이터 로드 ──────────────────────────────────────────
df = pd.read_csv(
    os.path.join(data_dir, "merged_multi.csv"),
    parse_dates=["date"], index_col="date"
)

# ── 2. 변수 정의 ────────────────────────────────────────────
y = df["mom_change_pct"]
X = df[["base_rate", "mortgage_rate", "csi", "housing_permit",
        "moon_dummy", "yoon_dummy"]]
X = sm.add_constant(X)

# ── 3. OLS 회귀분석 ─────────────────────────────────────────
model  = sm.OLS(y, X).fit()
print("=" * 60)
print("  다중회귀분석 결과")
print("=" * 60)
print(model.summary())

# ── 4. 단순회귀(기준금리만)와 R² 비교 ───────────────────────
X_simple = sm.add_constant(df[["base_rate"]])
r2_simple = sm.OLS(y, X_simple).fit().rsquared
print(f"\n[R² 비교]")
print(f"  단순회귀 (기준금리만) : {r2_simple:.4f}")
print(f"  다중회귀 (6개 변수)  : {model.rsquared:.4f}")
print(f"  R² 개선폭            : +{model.rsquared - r2_simple:.4f}")

# ── 5. VIF (다중공선성 진단) ────────────────────────────────
print("\n[VIF - 다중공선성 진단]")
print("  (VIF > 10 이면 다중공선성 심각)")
vif_data = pd.DataFrame()
vif_data["변수"] = X.columns
vif_data["VIF"]  = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
print(vif_data.to_string(index=False))

# ── 6. 계수 시각화 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 회귀계수 막대그래프 (상수항 제외)
coef = model.params.drop("const")
colors = ["tomato" if c < 0 else "steelblue" for c in coef]
axes[0].barh(coef.index, coef.values, color=colors)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("회귀계수 (β)")
axes[0].set_xlabel("계수 값")

# (b) 실제값 vs 예측값
axes[1].plot(df.index, y, label="실제값", alpha=0.7)
axes[1].plot(df.index, model.fittedvalues, label="예측값", alpha=0.7, linestyle="--")
axes[1].set_title("실제 집값 변동률 vs 예측값")
axes[1].set_ylabel("전월대비 변동률 (%)")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
out_path = os.path.join(fig_dir, "multi_regression.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장: {out_path}")
plt.show()

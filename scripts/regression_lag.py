"""
시차(Lag) 효과 분석 스크립트
==============================
주택 인허가에 6개월, 12개월, 24개월 시차를 적용하여
어느 시차가 집값 변동률을 가장 잘 설명하는지 비교합니다.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
for font in fm.fontManager.ttflist:
    if 'AppleGothic' in font.name or 'Malgun' in font.name:
        plt.rcParams['font.family'] = font.name
        break

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "..", "data")
fig_dir    = os.path.join(script_dir, "..", "figures")

# ── 1. 데이터 로드 ──────────────────────────────────────────
df = pd.read_csv(
    os.path.join(data_dir, "merged_multi.csv"),
    parse_dates=["date"], index_col="date"
)

# ── 2. 시차 변수 생성 ───────────────────────────────────────
df["permit_lag6"]  = df["housing_permit"].shift(6)
df["permit_lag12"] = df["housing_permit"].shift(12)
df["permit_lag24"] = df["housing_permit"].shift(24)

# ── 3. 시차별 모델 비교 ─────────────────────────────────────
base_vars = ["base_rate", "mortgage_rate", "csi", "moon_dummy", "yoon_dummy"]
lag_options = {
    "현재 (lag=0)":   "housing_permit",
    "6개월 (lag=6)":  "permit_lag6",
    "12개월 (lag=12)": "permit_lag12",
    "24개월 (lag=24)": "permit_lag24",
}

results = []
print("=" * 55)
print("  시차별 모델 비교")
print("=" * 55)
print(f"{'시차':<18} {'R²':>8} {'Adj.R²':>8} {'인허가 p값':>12} {'인허가 β':>10}")
print("-" * 55)

for label, permit_var in lag_options.items():
    sub = df[base_vars + [permit_var, "mom_change_pct"]].dropna()
    y = sub["mom_change_pct"]
    X = sm.add_constant(sub[base_vars + [permit_var]])
    m = sm.OLS(y, X).fit()

    pval  = m.pvalues[permit_var]
    coef  = m.params[permit_var]
    results.append({
        "label": label,
        "r2": m.rsquared,
        "adj_r2": m.rsquared_adj,
        "pval": pval,
        "coef": coef,
        "n": len(sub),
        "model": m,
        "permit_var": permit_var
    })
    sig = "★" if pval < 0.05 else ""
    print(f"{label:<18} {m.rsquared:>8.4f} {m.rsquared_adj:>8.4f} {pval:>12.4f} {coef:>10.6f} {sig}")

print("-" * 55)
print("★ p < 0.05 (통계적으로 유의)")

# ── 4. 최적 시차 모델 상세 출력 ────────────────────────────
best = max(results, key=lambda x: x["adj_r2"])
print(f"\n\n최적 시차: {best['label']} (Adj.R²={best['adj_r2']:.4f})")
print("=" * 60)
print(best["model"].summary())

# ── 5. R² 비교 시각화 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels  = [r["label"] for r in results]
r2_vals = [r["r2"] for r in results]
pvals   = [r["pval"] for r in results]

# (a) R² 막대그래프
colors = ["steelblue" if r["pval"] < 0.05 else "lightgray" for r in results]
axes[0].bar(labels, r2_vals, color=colors)
axes[0].set_title("시차별 R²\n(파란색 = 인허가 변수 p<0.05)")
axes[0].set_ylabel("R²")
axes[0].set_ylim(0, 0.6)
for i, v in enumerate(r2_vals):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

# (b) 최적 시차 모델 실제 vs 예측
best_sub = df[base_vars + [best["permit_var"], "mom_change_pct"]].dropna()
axes[1].plot(best_sub.index, best_sub["mom_change_pct"], label="실제값", alpha=0.7)
axes[1].plot(best_sub.index, best["model"].fittedvalues,
             label=f"예측값 ({best['label']})", alpha=0.7, linestyle="--")
axes[1].set_title(f"실제 vs 예측 ({best['label']})")
axes[1].set_ylabel("전월대비 변동률 (%)")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
out = os.path.join(fig_dir, "lag_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장: {out}")
plt.show()

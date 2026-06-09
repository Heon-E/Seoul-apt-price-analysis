"""
Newey-West (HAC) 표준오차 보정 스크립트
==========================================
OLS와 HAC 결과를 나란히 비교하여
자기상관 보정 전후 p값 변화를 확인합니다.
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

plt.rcParams['axes.unicode_minus'] = False
for font in fm.fontManager.ttflist:
    if 'AppleGothic' in font.name or 'Malgun' in font.name:
        plt.rcParams['font.family'] = font.name
        break

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "..", "data")
fig_dir    = os.path.join(script_dir, "..", "figures")

# ── 데이터 로드 ─────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(data_dir, "merged_multi.csv"),
    parse_dates=["date"], index_col="date"
)

y = df["mom_change_pct"]
X = sm.add_constant(df[["base_rate", "mortgage_rate", "csi",
                          "housing_permit", "moon_dummy", "yoon_dummy"]])

# ── OLS (보정 전) ────────────────────────────────────────────
ols = sm.OLS(y, X).fit()

# ── HAC (Newey-West 보정 후) ─────────────────────────────────
# maxlags=6 : 최대 6개월 전까지의 자기상관을 반영
hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

# ── 비교 출력 ────────────────────────────────────────────────
print("=" * 65)
print("  OLS vs Newey-West(HAC) 비교")
print("  (계수는 동일, 표준오차·p값만 달라집니다)")
print("=" * 65)
print(f"\n{'변수':<18} {'OLS β':>8} {'OLS p':>8} {'HAC p':>8} {'변화':>8}")
print("-" * 65)

vars_to_show = ["base_rate", "mortgage_rate", "csi",
                "housing_permit", "moon_dummy", "yoon_dummy"]

for v in vars_to_show:
    coef    = ols.params[v]
    p_ols   = ols.pvalues[v]
    p_hac   = hac.pvalues[v]

    # 유의성 변화 표시
    def sig(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    change = "→ 유의성 상실" if (p_ols < 0.05 and p_hac >= 0.05) else \
             "→ 유지" if (p_ols < 0.05 and p_hac < 0.05) else ""

    print(f"{v:<18} {coef:>8.4f} {p_ols:>7.4f}{sig(p_ols):1s} "
          f"{p_hac:>7.4f}{sig(p_hac):1s} {change}")

print("-" * 65)
print("*** p<0.01  ** p<0.05  * p<0.10")
print(f"\nR²: {ols.rsquared:.4f} (HAC는 계수 동일, R² 변화 없음)")
print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(ols.resid):.3f}")

# ── 시각화: p값 비교 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(vars_to_show))
width = 0.35

p_ols_vals = [ols.pvalues[v] for v in vars_to_show]
p_hac_vals = [hac.pvalues[v] for v in vars_to_show]

bars1 = ax.bar([i - width/2 for i in x], p_ols_vals, width,
               label="OLS (보정 전)", color="steelblue", alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], p_hac_vals, width,
               label="HAC (보정 후)", color="tomato", alpha=0.7)

ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="p=0.05 기준선")
ax.set_xticks(list(x))
ax.set_xticklabels(vars_to_show, rotation=20)
ax.set_ylabel("p값")
ax.set_title("OLS vs Newey-West HAC\n보정 전후 p값 비교")
ax.legend()
ax.set_ylim(0, 1.05)

plt.tight_layout()
out = os.path.join(fig_dir, "hac_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n그래프 저장: {out}")
plt.show()

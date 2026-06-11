"""
국면별 회귀분석 — 상승기 / 하락기 / 반등기
공급(준공 건수)의 효과가 국면에 따라 다른지 검증
"""
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")
import os

DATA_DIR = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/data"
)

merged = pd.read_csv(os.path.join(DATA_DIR, "merged_multi.csv"), parse_dates=["date"])
comp   = pd.read_csv(os.path.join(DATA_DIR, "house_completion.csv"), parse_dates=["date"])
df = merged.merge(comp, on="date", how="left").sort_values("date").dropna()

# 국면 정의
phases = {
    "상승기 (2015.01~2021.12)": ("2015-01", "2021-12"),
    "하락기 (2022.01~2023.06)": ("2022-01", "2023-06"),
    "반등기 (2023.07~2025.10)": ("2023-07", "2025-10"),
}

# 각 국면에서 사용할 변수 (더미가 구간 내 변동 없으면 제외)
phase_vars = {
    "상승기 (2015.01~2021.12)": ["mortgage_rate", "csi", "moon_dummy", "house_completion"],
    "하락기 (2022.01~2023.06)": ["mortgage_rate", "csi", "house_completion"],
    "반등기 (2023.07~2025.10)": ["mortgage_rate", "csi", "house_completion"],
}

print("=" * 70)
print("국면별 회귀분석 결과")
print("=" * 70)

for phase_name, (start, end) in phases.items():
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    vars_used = phase_vars[phase_name]

    X = add_constant(sub[vars_used])
    y = sub["mom_change_pct"]

    res = OLS(y, X).fit()
    dw  = durbin_watson(res.resid)

    print(f"\n▶ {phase_name}  (N={len(sub)})")
    print(f"  R² = {res.rsquared:.3f}  |  DW = {dw:.3f}")
    print(f"  {'변수':<20} {'계수':>10} {'p값':>8}")
    print(f"  {'-'*40}")
    for var in vars_used:
        coef = res.params[var]
        pval = res.pvalues[var]
        sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:<20} {coef:>10.4f} {pval:>8.3f} {sig}")

print("\n" + "=" * 70)
print("⚠ 하락기(N=18)는 관측치 부족으로 결과 불안정 — 참고용으로만 해석")
print("=" * 70)

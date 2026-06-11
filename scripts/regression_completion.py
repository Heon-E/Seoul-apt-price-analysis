"""
준공 건수 회귀분석 — 인허가 대비 공급 효과 비교
시차 0 / 6 / 12 / 24개월 비교
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

merged  = pd.read_csv(os.path.join(DATA_DIR, "merged_multi.csv"), parse_dates=["date"])
comp    = pd.read_csv(os.path.join(DATA_DIR, "house_completion.csv"), parse_dates=["date"])
df = merged.merge(comp, on="date", how="left").sort_values("date").dropna()

CONTROLS = ["mortgage_rate", "csi", "moon_dummy", "yoon_dummy"]
Y = df["mom_change_pct"]

print("=" * 65)
print("준공 건수 시차별 회귀분석 비교")
print("=" * 65)
print(f"{'시차':>6} | {'R²':>6} | {'준공 계수':>10} | {'p값':>7} | {'DW':>6}")
print("-" * 65)

for lag in [0, 6, 12, 24]:
    df[f"completion_lag{lag}"] = df["house_completion"].shift(lag)
    sub = df.dropna(subset=[f"completion_lag{lag}"])

    X = add_constant(sub[CONTROLS + [f"completion_lag{lag}"]])
    y = sub["mom_change_pct"]

    res = OLS(y, X).fit()
    coef = res.params[f"completion_lag{lag}"]
    pval = res.pvalues[f"completion_lag{lag}"]
    dw   = durbin_watson(res.resid)
    sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))

    print(f"{lag:>6}개월 | {res.rsquared:>6.3f} | {coef:>10.6f} | {pval:>7.3f} {sig:<3} | {dw:>6.3f}")

print("=" * 65)
print("\n[참고] 인허가 기존 결과:")
print(f"{'시차':>6} | {'R²':>6} | {'인허가 계수':>11} | {'p값':>7}")
print("-" * 50)
for lag in [0, 6, 12, 24]:
    df[f"permit_lag{lag}"] = df["housing_permit"].shift(lag)
    sub = df.dropna(subset=[f"permit_lag{lag}"])
    X = add_constant(sub[CONTROLS + [f"permit_lag{lag}"]])
    y = sub["mom_change_pct"]
    res = OLS(y, X).fit()
    coef = res.params[f"permit_lag{lag}"]
    pval = res.pvalues[f"permit_lag{lag}"]
    print(f"{lag:>6}개월 | {res.rsquared:>6.3f} | {coef:>11.6f} | {pval:>7.3f}")

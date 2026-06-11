import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math

# ─── 1. 데이터 구성 ───────────────────────────────────────────

# BOJ 기준금리 (Basic Loan Rate, 2010/01~2024/12)
# 2010~2024/07: 0.3%, 2024/08~: 0.5%
dates = pd.date_range("2010-01", periods=180, freq="MS")
boj_rate = [0.3]*175 + [0.5]*5  # 175 = 2010/01~2024/07

# USD/JPY 월평균 환율
usdjpy = [
  91.26,90.28,90.56,93.43,91.79,90.89,87.67,85.44,84.31,81.80,82.43,83.38,  # 2010
  82.63,82.52,81.82,83.34,81.23,80.49,79.44,77.09,76.78,76.72,77.50,77.81,  # 2011
  76.94,78.47,82.37,81.42,79.70,79.27,78.96,78.68,78.17,78.97,80.92,83.60,  # 2012
  89.15,93.07,94.73,97.74,101.01,97.52,99.66,97.83,99.30,97.73,100.04,103.42, # 2013
  103.94,102.02,102.30,102.54,101.78,102.05,101.73,102.95,107.16,108.03,116.24,119.29, # 2014
  118.25,118.59,120.37,119.57,120.82,123.70,123.31,123.17,120.13,119.99,122.58,121.78, # 2015
  118.18,115.01,113.05,109.72,109.24,105.44,103.97,101.28,101.99,103.81,108.33,116.01, # 2016
  114.69,113.13,113.02,110.08,112.24,110.89,112.50,109.90,110.67,112.94,112.89,112.96, # 2017
  110.74,107.90,106.01,107.49,109.74,110.02,111.41,111.06,111.91,112.81,113.36,112.38, # 2018
  108.97,110.36,111.22,111.63,109.76,108.07,108.23,106.34,107.40,108.12,108.88,109.18, # 2019
  109.38,109.96,107.42,107.85,107.28,107.60,106.75,106.02,105.67,105.21,104.40,103.83, # 2020
  103.70,105.38,108.70,109.10,109.13,110.09,110.26,109.82,110.20,113.09,114.03,113.88, # 2021
  114.84,115.16,118.54,126.13,128.68,133.85,136.70,135.28,143.09,147.16,142.17,134.85, # 2022
  130.28,132.69,133.86,133.40,137.39,141.33,141.20,144.73,147.65,149.59,149.88,144.09, # 2023
  146.59,149.41,149.70,153.57,156.21,157.90,157.86,146.29,143.31,149.60,153.82,153.72  # 2024
]

# 맨션 가격지수 (MLIT, 도쿄, 2010=100)
mansion_raw = pd.read_csv(
    "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/data/tokyo_mansion_index.csv"
)

df = pd.DataFrame({
    'date': dates,
    'mansion_idx': mansion_raw['mansion_index'].values,
    'boj_rate': boj_rate,
    'usdjpy': usdjpy
})
df.set_index('date', inplace=True)

# ─── 2. 종속변수: 전월대비 변화율 (MoM%) ──────────────────────
df['mom_pct'] = df['mansion_idx'].pct_change() * 100
df = df.dropna()

print("=== 데이터 기본 통계 ===")
print(df[['mom_pct','boj_rate','usdjpy']].describe().round(2))
print(f"\n기간: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
print(f"관측치: {len(df)}개")

# ─── 3. ARIMAX 추정 함수 ────────────────────────────────────

def t_sf(t, df_):
    """Two-tailed p-value (normal approx, fine for df>30)"""
    import math
    return math.erfc(abs(t) / math.sqrt(2))

def fit_arimax(exog_cols, data, y_col='mom_pct'):
    cols = exog_cols + [y_col]
    valid = data[cols].dropna()
    y_t   = valid[y_col].values[1:]
    y_lag = valid[y_col].values[:-1]
    X = np.column_stack([np.ones(len(y_t)), y_lag] +
                        [valid[c].values[1:] for c in exog_cols])
    beta, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)
    resid = y_t - X @ beta
    n, k  = len(y_t), len(beta)
    sigma2 = np.sum(resid**2) / n
    ll  = -n/2 * np.log(2*np.pi*sigma2) - np.sum(resid**2)/(2*sigma2)
    aic = -2*ll + 2*k
    bic = -2*ll + k*np.log(n)
    dw  = np.sum(np.diff(resid)**2) / np.sum(resid**2)
    s2  = np.sum(resid**2) / (n-k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se  = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = np.array([t_sf(abs(t), n-k) for t in tvals])
    fitted = X @ beta
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_t - y_t.mean())**2)
    r2 = 1 - ss_res/ss_tot
    labels = ['const','AR(1)'] + exog_cols
    return {
        'N':n,'AIC':aic,'BIC':bic,'DW':dw,'R2':r2,
        'beta':beta,'se':se,'tvals':tvals,'pvals':pvals,
        'labels':labels,'fitted':fitted,'y':y_t,'resid':resid,
        'dates': valid.index[1:]
    }

# ─── 4. 모형 비교 ─────────────────────────────────────────────
models = {
    'M1 (AR only)':         [],
    'M2 (+BOJ금리)':        ['boj_rate'],
    'M3 (+USD/JPY)':        ['usdjpy'],
    'M4 (+금리+환율)':      ['boj_rate','usdjpy'],
}

results = {}
print("\n=== 모형 비교 ===")
print(f"{'모형':<18} {'N':>4} {'AIC':>8} {'BIC':>8} {'DW':>6} {'R²':>6}")
print("-"*52)
for name, exog in models.items():
    r = fit_arimax(exog, df)
    results[name] = r
    print(f"{name:<18} {r['N']:>4} {r['AIC']:>8.2f} {r['BIC']:>8.2f} {r['DW']:>6.3f} {r['R2']:>6.3f}")

# ─── 5. 최적 모형 계수 출력 ────────────────────────────────────
best = results['M4 (+금리+환율)']
print("\n=== M4 계수 상세 (금리 + 환율) ===")
print(f"{'변수':<12} {'계수':>8} {'표준오차':>10} {'t값':>8} {'p값':>8} {'유의성':>6}")
print("-"*58)
for i, lbl in enumerate(best['labels']):
    sig = '***' if best['pvals'][i]<0.01 else ('**' if best['pvals'][i]<0.05 else ('*' if best['pvals'][i]<0.1 else ''))
    print(f"{lbl:<12} {best['beta'][i]:>8.4f} {best['se'][i]:>10.4f} {best['tvals'][i]:>8.3f} {best['pvals'][i]:>8.4f} {sig:>6}")

# ─── 6. 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle('Tokyo Mansion Price Analysis — ARIMAX(1,0,0)\n(MLIT Price Index, BOJ Rate, USD/JPY  |  2010–2024)',
             fontsize=13, fontweight='bold')

# 패널 1: 맨션 지수 + 환율
ax1 = axes[0]
ax1b = ax1.twinx()
ax1.plot(df.index, df['mansion_idx'], color='#1f77b4', lw=2, label='Tokyo Mansion Index (LHS)')
ax1b.plot(df.index, df['usdjpy'], color='#d62728', lw=1.5, alpha=0.8, ls='--', label='USD/JPY (RHS)')
ax1.set_ylabel('Price Index (2010=100)', color='#1f77b4')
ax1b.set_ylabel('USD/JPY', color='#d62728')
ax1.set_title('① 도쿄 맨션 가격지수 vs 엔화 환율', fontsize=11)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc='upper left', fontsize=9)

# 패널 2: Actual vs Fitted (M4)
ax2 = axes[1]
ax2.plot(best['dates'], best['y'], color='gray', lw=1.5, label='Actual MoM%', alpha=0.7)
ax2.plot(best['dates'], best['fitted'], color='#1f77b4', lw=2, label='M4 Fitted')
ax2.axhline(0, color='black', lw=0.8, ls=':')
ax2.set_ylabel('MoM Change (%)')
ax2.set_title(f'② Actual vs Fitted  |  M4: R²={best["R2"]:.3f}, AIC={best["AIC"]:.1f}, DW={best["DW"]:.2f}', fontsize=11)
ax2.legend(fontsize=9)

# 패널 3: AIC/BIC 비교
ax3 = axes[2]
model_names = list(results.keys())
aics = [results[m]['AIC'] for m in model_names]
bics = [results[m]['BIC'] for m in model_names]
x = np.arange(len(model_names))
bars1 = ax3.bar(x - 0.2, aics, 0.4, label='AIC', color='#1f77b4', alpha=0.8)
bars2 = ax3.bar(x + 0.2, bics, 0.4, label='BIC', color='#ff7f0e', alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, fontsize=9)
ax3.set_ylabel('Information Criterion')
ax3.set_title('③ 모형별 AIC/BIC 비교  (낮을수록 좋음)', fontsize=11)
ax3.legend(fontsize=9)
for bar in bars1: ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2: ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig.savefig(
    "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/figures/tokyo_arimax.png",
    dpi=150, bbox_inches='tight'
)
print("\n그래프 저장 완료: tokyo_arimax.png")
print("\n=== 완료 ===")

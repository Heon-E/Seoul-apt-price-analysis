import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

# ─── 0. 데이터 ──────────────────────────────────────────────
BASE = "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/data"
dates  = pd.date_range("2010-01", periods=180, freq="MS")
mansion = pd.read_csv(f"{BASE}/tokyo_mansion_index.csv")['mansion_index'].values
mb_raw  = pd.read_csv(f"{BASE}/boj_monetary_base.csv")['mb_level_100m_yen'].values
mb_yoy  = pd.read_csv(f"{BASE}/boj_monetary_base.csv")['mb_yoy_pct'].values
ppi_yoy = pd.read_csv(f"{BASE}/japan_ppi_yoy.csv")['ppi_yoy'].values

# PPI 수준 (2020=100 기준)
ppi_level_raw = [
  97.1,97.1,97.1,97.4,97.4,97.1,97.1,97.0,97.0,96.9,96.8,97.3,
  97.7,97.8,98.3,99.1,98.9,98.9,99.2,99.1,98.9,98.1,98.0,98.0,
  97.9,98.1,98.6,98.4,98.0,97.5,97.1,97.2,97.5,97.1,97.0,97.4,
  97.6,98.0,98.1,98.5,98.6,98.6,99.1,99.4,99.7,99.5,99.5,99.8,
  100.0,99.8,99.8,102.7,103.0,103.1,103.5,103.4,103.3,102.3,102.1,101.6,
  100.3,100.3,100.5,100.5,100.7,100.6,100.2,99.5,99.1,98.4,98.4,98.0,
  96.9,96.6,96.5,96.1,96.1,96.1,96.0,95.7,95.8,95.7,96.1,96.8,
  97.4,97.6,97.9,98.1,98.1,98.2,98.4,98.4,98.7,99.1,99.5,99.7,
  100.0,100.0,99.9,100.3,100.8,101.1,101.5,101.5,101.7,102.1,101.7,101.1,
  100.5,100.9,101.2,101.6,101.4,100.9,100.8,100.6,100.6,101.7,101.8,102.0,
  102.1,101.7,100.8,99.2,98.7,99.3,99.8,99.9,99.8,99.5,99.4,99.8,
  100.3,100.8,101.8,103.0,103.6,104.3,105.4,105.6,106.0,107.7,108.4,108.4,
  109.5,110.4,111.5,113.5,113.5,114.5,115.4,115.9,117.0,118.2,119.2,119.9,
  120.0,119.7,119.8,120.4,119.6,119.6,119.7,120.0,119.8,119.6,119.9,120.2,
  120.4,120.6,121.0,121.8,122.7,123.0,123.6,123.2,123.6,124.2,124.6,125.1
]
ppi_level = np.array(ppi_level_raw)

# BOJ 정책금리 (콜금리) - 월별
boj_rate = np.where(
    np.arange(180) < 73,  # 2010/01-2016/01 → 0.1%
    0.1,
    np.where(np.arange(180) < 170, -0.1,  # 2016/02-2024/02 → -0.1%
    np.where(np.arange(180) < 174, 0.1, 0.25))  # 2024/03~06: 0.1%, 07~: 0.25%
)

# 실질금리 = 명목금리 - PPI YoY
real_rate = boj_rate - ppi_yoy  # 단위: %

usdjpy = np.array([
  91.26,90.28,90.56,93.43,91.79,90.89,87.67,85.44,84.31,81.80,82.43,83.38,
  82.63,82.52,81.82,83.34,81.23,80.49,79.44,77.09,76.78,76.72,77.50,77.81,
  76.94,78.47,82.37,81.42,79.70,79.27,78.96,78.68,78.17,78.97,80.92,83.60,
  89.15,93.07,94.73,97.74,101.01,97.52,99.66,97.83,99.30,97.73,100.04,103.42,
  103.94,102.02,102.30,102.54,101.78,102.05,101.73,102.95,107.16,108.03,116.24,119.29,
  118.25,118.59,120.37,119.57,120.82,123.70,123.31,123.17,120.13,119.99,122.58,121.78,
  118.18,115.01,113.05,109.72,109.24,105.44,103.97,101.28,101.99,103.81,108.33,116.01,
  114.69,113.13,113.02,110.08,112.24,110.89,112.50,109.90,110.67,112.94,112.89,112.96,
  110.74,107.90,106.01,107.49,109.74,110.02,111.41,111.06,111.91,112.81,113.36,112.38,
  108.97,110.36,111.22,111.63,109.76,108.07,108.23,106.34,107.40,108.12,108.88,109.18,
  109.38,109.96,107.42,107.85,107.28,107.60,106.75,106.02,105.67,105.21,104.40,103.83,
  103.70,105.38,108.70,109.10,109.13,110.09,110.26,109.82,110.20,113.09,114.03,113.88,
  114.84,115.16,118.54,126.13,128.68,133.85,136.70,135.28,143.09,147.16,142.17,134.85,
  130.28,132.69,133.86,133.40,137.39,141.33,141.20,144.73,147.65,149.59,149.88,144.09,
  146.59,149.41,149.70,153.57,156.21,157.90,157.86,146.29,143.31,149.60,153.82,153.72
])

log_mansion = np.log(mansion)
log_mb      = np.log(mb_raw)
log_ppi     = np.log(ppi_level)

# ─── ADF 함수 ───────────────────────────────────────────────
def adf_test(series, max_lags=4, label='', critical=-2.86, verbose=True):
    y = np.array(series, dtype=float)
    dy = np.diff(y)
    best_aic, best_tstat = np.inf, None
    for lags in range(0, max_lags+1):
        n = len(dy) - lags
        if n < 10: continue
        y_lag = y[lags:lags+n]
        diffs = [dy[lags-j-1:lags-j-1+n] for j in range(lags)]
        X = np.column_stack([np.ones(n), y_lag] + diffs)
        Y = dy[lags:lags+n]
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ b
        s2 = np.sum(resid**2)/(n - X.shape[1])
        ll = -n/2*np.log(2*np.pi*s2) - np.sum(resid**2)/(2*s2)
        aic = -2*ll + 2*X.shape[1]
        if aic < best_aic:
            best_aic = aic
            cov = s2 * np.linalg.inv(X.T @ X)
            best_tstat = b[1] / np.sqrt(np.diag(cov)[1])
    result = 'STATIONARY' if best_tstat < critical else 'UNIT ROOT'
    if verbose:
        print(f"  {label:<25} t={best_tstat:>7.3f}  → {result}")
    return best_tstat

def t_sf(t):
    return math.erfc(abs(t)/math.sqrt(2))

# ─── 1. 신규 변수 단위근 검정 ─────────────────────────────
print("=" * 60)
print("ADF: 신규 변수 단위근 검정")
print(f"  (임계값 -2.86)")
adf_test(log_ppi,   label='log(PPI level)')
adf_test(np.diff(log_ppi), label='Δlog(PPI level)')
adf_test(real_rate, label='실질금리 (명목-PPI YoY)')
adf_test(np.diff(real_rate), label='Δ실질금리')

# ─── 2. 다변수 Engle-Granger 공적분 검정 ───────────────────
print("\n" + "=" * 60)
print("Engle-Granger 공적분 검정 — 여러 조합")
print("  (잔차 ADF 임계값 ≈ -3.34 (2변수), -3.74 (3변수))")

def engle_granger_test(y, X_list, labels):
    n = len(y)
    X = np.column_stack([np.ones(n)] + X_list)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # 잔차 ADF (lags=0만 보고, 1,2도 체크)
    t0 = adf_test(resid, max_lags=0, label='', critical=-9999, verbose=False)
    t1 = adf_test(resid, max_lags=1, label='', critical=-9999, verbose=False)
    k = len(X_list)
    crit = -3.34 if k == 1 else (-3.74 if k == 2 else -4.10)
    cointegrated = t0 < crit
    eqn = "log(mansion) = {:.4f}".format(beta[0])
    for i, lbl in enumerate(labels):
        eqn += " + {:.4f}·{}".format(beta[i+1], lbl)
    print(f"\n  {' + '.join(labels)}")
    print(f"    {eqn}")
    print(f"    잔차 ADF t={t0:.3f} (lags=0), t={t1:.3f} (lags=1)  |  임계값 {crit}")
    print(f"    → {'✓ COINTEGRATED' if cointegrated else '✗ NOT cointegrated'}")
    return resid, beta, cointegrated

combos = [
    ([log_mb],                   ['log(MB)']),
    ([log_ppi],                  ['log(PPI)']),
    ([usdjpy],                   ['USD/JPY']),
    ([log_mb, usdjpy],           ['log(MB)', 'USD/JPY']),
    ([log_mb, log_ppi],          ['log(MB)', 'log(PPI)']),
    ([log_ppi, usdjpy],          ['log(PPI)', 'USD/JPY']),
    ([log_mb, log_ppi, usdjpy],  ['log(MB)', 'log(PPI)', 'USD/JPY']),
]

best_ect, best_beta, best_labels = None, None, None
best_t = 0
for X_list, labels in combos:
    resid, beta, cointegrated = engle_granger_test(log_mansion, X_list, labels)
    t_val = adf_test(resid, max_lags=0, label='', critical=-9999, verbose=False)
    if t_val < best_t:
        best_t = t_val
        best_ect, best_beta, best_labels = resid, beta, labels

print(f"\n  ▶ 가장 강한 공적분 후보: {best_labels}  (t={best_t:.3f})")

# ─── 3. 최적 조합으로 ECM ───────────────────────────────────
print("\n" + "=" * 60)
print(f"ECM — 최적 변수 조합으로 추정")

# ARIMAX MoM% 모델에도 실질금리 추가 시도
d_lm  = np.diff(log_mansion)
d_lmb = np.diff(log_mb)
d_lppi = np.diff(log_ppi)
d_fx  = np.diff(usdjpy)
d_rr  = np.diff(real_rate)   # Δ실질금리
ect_v1 = best_ect[:-1] if best_ect is not None else np.zeros(len(d_lm))

# 여러 ECM 스펙 비교
def fit_ecm(y, X_cols_data, col_names):
    n = len(y)
    X = np.column_stack([np.ones(n)] + X_cols_data)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = X.shape[1]
    s2 = np.sum(resid**2)/(n-k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = np.array([t_sf(t) for t in tvals])
    sigma2 = np.sum(resid**2)/n
    ll = -n/2*np.log(2*np.pi*sigma2) - np.sum(resid**2)/(2*sigma2)
    aic = -2*ll + 2*k
    r2 = 1 - np.sum(resid**2)/np.sum((y-y.mean())**2)
    dw = np.sum(np.diff(resid)**2)/np.sum(resid**2)
    return dict(beta=beta, se=se, tvals=tvals, pvals=pvals,
                aic=aic, r2=r2, dw=dw, resid=resid,
                fitted=X@beta, names=['const']+col_names)

specs = [
    ([ect_v1, d_lmb, d_fx],      ['ECT(-1)', 'Δlog(MB)', 'ΔUSD/JPY'],       'ECM v1 (MB+FX)'),
    ([ect_v1, d_lmb, d_lppi],    ['ECT(-1)', 'Δlog(MB)', 'Δlog(PPI)'],       'ECM v2 (MB+PPI)'),
    ([ect_v1, d_lmb, d_fx, d_rr],['ECT(-1)', 'Δlog(MB)', 'ΔUSD/JPY', 'Δ실질금리'],'ECM v3 (+실질금리)'),
    ([d_lmb, d_fx, d_rr],        ['Δlog(MB)', 'ΔUSD/JPY', 'Δ실질금리'],       'ARIMAX+실질금리'),
    ([d_lppi, d_rr],             ['Δlog(PPI)', 'Δ실질금리'],                   'ARIMAX PPI+실질금리'),
]

print(f"\n  {'모델':<25} {'AIC':>8} {'R²':>7} {'DW':>6}  주요 유의변수")
print("  " + "-"*65)
all_fits = {}
for data, names, label in specs:
    fit = fit_ecm(d_lm, data, names)
    sig_vars = [names[i] for i in range(len(names)) if fit['pvals'][i+1] < 0.1]
    print(f"  {label:<25} {fit['aic']:>8.2f} {fit['r2']:>7.3f} {fit['dw']:>6.3f}  {', '.join(sig_vars) if sig_vars else '없음'}")
    all_fits[label] = fit

# 상세 출력: 실질금리 모델
print(f"\n  ── 상세: ARIMAX + 실질금리 ──")
fit = all_fits['ARIMAX+실질금리']
print(f"  N={len(d_lm)}, AIC={fit['aic']:.2f}, R²={fit['r2']:.3f}")
for i, nm in enumerate(fit['names']):
    sig = '***' if fit['pvals'][i]<0.01 else ('**' if fit['pvals'][i]<0.05 else ('*' if fit['pvals'][i]<0.1 else ''))
    print(f"    {nm:<15} β={fit['beta'][i]:>8.4f}  t={fit['tvals'][i]:>6.3f}  p={fit['pvals'][i]:>6.4f}  {sig}")

print(f"\n  ── 상세: ECM v2 (MB + PPI) ──")
fit2 = all_fits['ECM v2 (MB+PPI)']
print(f"  N={len(d_lm)}, AIC={fit2['aic']:.2f}, R²={fit2['r2']:.3f}")
for i, nm in enumerate(fit2['names']):
    sig = '***' if fit2['pvals'][i]<0.01 else ('**' if fit2['pvals'][i]<0.05 else ('*' if fit2['pvals'][i]<0.1 else ''))
    print(f"    {nm:<15} β={fit2['beta'][i]:>8.4f}  t={fit2['tvals'][i]:>6.3f}  p={fit2['pvals'][i]:>6.4f}  {sig}")

# ─── 4. 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Tokyo Mansion — Extended Cointegration & Variable Analysis\n2010–2024',
             fontsize=12, fontweight='bold')
t_idx = pd.date_range("2010-02", periods=len(d_lm), freq="MS")

# 패널1: 수준 시리즈 비교
ax = axes[0,0]
ax2 = ax.twinx()
ax.plot(dates, log_mansion, 'b-', lw=2, label='log(맨션지수) L')
ax.plot(dates, log_ppi, 'g--', lw=1.5, label='log(PPI) L')
ax2.plot(dates, log_mb, 'r:', lw=1.5, label='log(MB) R')
ax.set_ylabel('log(index)', color='b')
ax2.set_ylabel('log(MB)', color='r')
ax.set_title('Level Series: 맨션 vs PPI vs MB')
l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,lb1+lb2,loc='upper left',fontsize=8)

# 패널2: 실질금리 추이
ax = axes[0,1]
ax.fill_between(dates, real_rate, 0,
                where=real_rate < 0, color='red', alpha=0.3, label='음의 실질금리')
ax.fill_between(dates, real_rate, 0,
                where=real_rate >= 0, color='blue', alpha=0.3, label='양의 실질금리')
ax.plot(dates, real_rate, 'k-', lw=1.2)
ax.axhline(0, color='gray', lw=0.7, ls='--')
ax.set_ylabel('% (명목금리 - PPI YoY)')
ax.set_title('실질금리 추이 (BOJ rate - PPI YoY%)')
ax.legend(fontsize=9)

# 패널3: AIC 비교
ax = axes[1,0]
labels_plot = list(all_fits.keys())
aics = [all_fits[l]['aic'] for l in labels_plot]
r2s  = [all_fits[l]['r2'] for l in labels_plot]
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
bars = ax.bar(range(len(labels_plot)), aics, color=colors, alpha=0.8)
ax.set_xticks(range(len(labels_plot)))
ax.set_xticklabels([l.replace(' ','\n') for l in labels_plot], fontsize=7)
ax.set_ylabel('AIC (낮을수록 좋음)')
ax.set_title('모델 AIC 비교')
for bar, r in zip(bars, r2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'R²={r:.3f}', ha='center', va='bottom', fontsize=7)

# 패널4: 최고 모델 Actual vs Fitted
best_model_name = min(all_fits, key=lambda x: all_fits[x]['aic'])
fit_best = all_fits[best_model_name]
ax = axes[1,1]
ax.plot(t_idx, d_lm*100, 'gray', lw=1.5, alpha=0.7, label='Actual Δlog×100')
ax.plot(t_idx, fit_best['fitted']*100, 'r-', lw=2, label=f'Fitted ({best_model_name})')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_ylabel('월간변화(%)')
ax.set_title(f'Best Model: {best_model_name}\nAIC={fit_best["aic"]:.1f}, R²={fit_best["r2"]:.3f}')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(
    "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/figures/tokyo_ecm_v2.png",
    dpi=150, bbox_inches='tight')
print("\n그래프 저장: tokyo_ecm_v2.png")

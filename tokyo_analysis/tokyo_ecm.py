import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 0. 데이터 로드 ──────────────────────────────────────────
BASE = "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/data"
dates  = pd.date_range("2010-01", periods=180, freq="MS")
mansion = pd.read_csv(f"{BASE}/tokyo_mansion_index.csv")['mansion_index'].values
mb_raw  = pd.read_csv(f"{BASE}/boj_monetary_base.csv")['mb_level_100m_yen'].values
mb_yoy  = pd.read_csv(f"{BASE}/boj_monetary_base.csv")['mb_yoy_pct'].values
usdjpy  = np.array([
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

# log 변환 (비율 스케일 맞춤)
log_mansion = np.log(mansion)
log_mb      = np.log(mb_raw)

# ─── 1. ADF 단위근 검정 ──────────────────────────────────────
def adf_test(series, max_lags=4, label=''):
    """ADF with constant, automatic lag by AIC"""
    y = np.array(series)
    dy = np.diff(y)
    best_aic, best_res = np.inf, None

    for lags in range(0, max_lags+1):
        n = len(dy) - lags
        y_lag = y[lags:-1] if lags < len(dy)-1 else y[:-1]
        y_lag = y_lag[-n:]
        diffs = [dy[lags-j-1:lags-j-1+n] for j in range(lags)]
        X = [np.ones(n), y_lag] + diffs
        X = np.column_stack(X)
        Y = dy[lags:lags+n]
        b, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ b
        s2 = np.sum(resid**2) / (n - X.shape[1])
        ll = -n/2 * np.log(2*np.pi*s2) - np.sum(resid**2)/(2*s2)
        aic = -2*ll + 2*X.shape[1]
        if aic < best_aic:
            best_aic = aic
            cov = s2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            tstat = b[1] / se[1]
            best_res = (tstat, lags, n)

    tstat, lags, n = best_res
    # MacKinnon 5% critical value (constant only): -2.86
    result = 'REJECT H0 (stationary)' if tstat < -2.86 else 'FAIL TO REJECT H0 (unit root)'
    print(f"  {label:<25} t={tstat:>7.3f}  lags={lags}  → {result}")
    return tstat

print("=== 1단계: ADF 단위근 검정 (H0: 단위근 존재) ===")
print("  (5% 임계값: -2.86)")
adf_test(log_mansion, label='log(맨션지수)')
adf_test(log_mb,      label='log(본원통화)')
adf_test(usdjpy,      label='USD/JPY')
adf_test(np.diff(log_mansion), label='Δlog(맨션지수)')
adf_test(np.diff(log_mb),      label='Δlog(본원통화)')
adf_test(np.diff(usdjpy),      label='ΔUSD/JPY')

# ─── 2. Engle-Granger 공적분 검정 ───────────────────────────
print("\n=== 2단계: Engle-Granger 공적분 검정 ===")
print("  (잔차의 단위근 검정: 5% 임계값 ≈ -3.34)")

def engle_granger(y, X_cols, X_data, labels):
    # 장기 수준 회귀
    n = len(y)
    X = np.column_stack([np.ones(n)] + [X_data[c] for c in X_cols])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    print(f"\n  장기 회귀: log(mansion) = {beta[0]:.4f}", end='')
    for i, lbl in enumerate(labels):
        print(f" + {beta[i+1]:.4f}·{lbl}", end='')
    print()

    # 잔차 ADF (공적분 잔차의 임계값은 더 엄격: ~-3.34)
    dy = np.diff(resid)
    y_lag = resid[:-1]
    for lags in [0, 1, 2]:
        n2 = len(dy) - lags
        y_l = y_lag[lags:lags+n2]
        diffs = [dy[lags-j-1:lags-j-1+n2] for j in range(lags)]
        Xr = [np.ones(n2), y_l] + diffs
        Xr = np.column_stack(Xr)
        Yr = dy[lags:lags+n2]
        b, _, _, _ = np.linalg.lstsq(Xr, Yr, rcond=None)
        s2 = np.sum((Yr - Xr@b)**2) / (n2 - Xr.shape[1])
        cov = s2 * np.linalg.inv(Xr.T @ Xr)
        t = b[1] / np.sqrt(np.diag(cov)[1])
        if lags == 0:
            result = 'COINTEGRATED ✓' if t < -3.34 else 'NOT cointegrated'
            print(f"  잔차 ADF t={t:.3f} (lags=0) → {result}")
    return resid, beta

data_dict = {'log_mb': log_mb, 'usdjpy': usdjpy}
ect_resid, lr_beta = engle_granger(
    log_mansion, ['log_mb','usdjpy'], data_dict,
    ['log(MB)', 'USD/JPY']
)

# ─── 3. ECM 추정 ─────────────────────────────────────────────
print("\n=== 3단계: ECM (Error Correction Model) ===")
print("  Δlog(mansion)_t = α + γ·ECT_{t-1} + β₁·Δlog(MB)_t + β₂·ΔUSD/JPY_t + ε_t")

def t_sf(t):
    return math.erfc(abs(t) / math.sqrt(2))

# 변수 구성
d_lm  = np.diff(log_mansion)       # Δlog(mansion) ≈ MoM%/100
d_lmb = np.diff(log_mb)            # Δlog(MB)
d_fx  = np.diff(usdjpy)            # Δ USD/JPY
ect   = ect_resid[:-1]             # ECT_{t-1} (lagged)

n = len(d_lm)
X = np.column_stack([np.ones(n), ect, d_lmb, d_fx])
beta, _, _, _ = np.linalg.lstsq(X, d_lm, rcond=None)
resid = d_lm - X @ beta
k = X.shape[1]
s2 = np.sum(resid**2) / (n - k)
cov = s2 * np.linalg.inv(X.T @ X)
se  = np.sqrt(np.diag(cov))
tvals = beta / se
pvals = np.array([t_sf(t) for t in tvals])
fitted = X @ beta

sigma2 = np.sum(resid**2) / n
ll  = -n/2*np.log(2*np.pi*sigma2) - np.sum(resid**2)/(2*sigma2)
aic = -2*ll + 2*k
r2  = 1 - np.sum(resid**2)/np.sum((d_lm - d_lm.mean())**2)
dw  = np.sum(np.diff(resid)**2)/np.sum(resid**2)

labels = ['const', 'ECT(-1)', 'Δlog(MB)', 'ΔUSD/JPY']
print(f"\n  N={n}, AIC={aic:.2f}, R²={r2:.3f}, DW={dw:.3f}")
print(f"\n  {'변수':<12} {'계수':>8} {'표준오차':>10} {'t값':>8} {'p값':>8} {'유의성':>5}")
print("  " + "-"*56)
for i, lbl in enumerate(labels):
    sig = '***' if pvals[i]<0.01 else ('**' if pvals[i]<0.05 else ('*' if pvals[i]<0.1 else ''))
    print(f"  {lbl:<12} {beta[i]:>8.4f} {se[i]:>10.4f} {tvals[i]:>8.3f} {pvals[i]:>8.4f} {sig:>5}")

print(f"\n  ▶ ECT 계수(γ) = {beta[1]:.4f}")
print(f"    해석: 균형에서 1단위 이탈 시 매월 {abs(beta[1])*100:.1f}% 수렴")
if beta[1] < 0:
    half_life = -np.log(2) / np.log(1 + beta[1])
    print(f"    반감기: {half_life:.1f}개월 ({half_life/12:.1f}년)")

# ─── 4. 시각화 ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle('Tokyo Mansion ECM — Cointegration + Error Correction\n'
             'log(MB) + USD/JPY  |  2010–2024', fontsize=12, fontweight='bold')

t_idx = pd.date_range("2010-02", periods=n, freq="MS")

# 패널1: 장기균형 관계
ax1 = axes[0]
ax1b = ax1.twinx()
ax1.plot(dates, log_mansion, color='#1f77b4', lw=2, label='log(Mansion Index)')
ax1.plot(dates, lr_beta[0] + lr_beta[1]*log_mb + lr_beta[2]*usdjpy,
         color='#d62728', lw=1.5, ls='--', label='Long-run Equilibrium')
ax1b.plot(dates[1:], ect_resid[1:]*100, color='#9467bd', lw=1, alpha=0.7, label='ECT (×100, RHS)')
ax1b.axhline(0, color='gray', lw=0.7, ls=':')
ax1.set_ylabel('log(Price Index)', color='#1f77b4')
ax1b.set_ylabel('Deviation from Equilibrium', color='#9467bd')
ax1.set_title('Panel 1: Long-run Equilibrium & Error Correction Term', fontsize=11)
l1,lb1 = ax1.get_legend_handles_labels()
l2,lb2 = ax1b.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)

# 패널2: Actual vs ECM Fitted
ax2 = axes[1]
ax2.plot(t_idx, d_lm*100, color='gray', lw=1.5, alpha=0.7, label='Actual Δlog×100 (≈MoM%)')
ax2.plot(t_idx, fitted*100, color='#d62728', lw=2, label=f'ECM Fitted (R²={r2:.3f})')
ax2.axhline(0, color='black', lw=0.7, ls=':')
ax2.set_ylabel('Monthly Change (%)')
ax2.set_title(f'Panel 2: Actual vs ECM Fitted  |  AIC={aic:.1f}, DW={dw:.2f}', fontsize=11)
ax2.legend(fontsize=9)

# 패널3: ECM 계수 시각화
ax3 = axes[2]
coef_labels = ['ECT(-1)', 'Δlog(MB)', 'ΔUSD/JPY']
coefs  = [beta[1], beta[2], beta[3]]
errors = [se[1]*1.96, se[2]*1.96, se[3]*1.96]
cols   = ['#d62728' if pvals[i+1]<0.05 else '#aec7e8' for i in range(3)]
y_pos  = np.arange(3)
ax3.barh(y_pos, coefs, xerr=errors, color=cols, alpha=0.85,
         error_kw=dict(ecolor='black', capsize=5))
ax3.axvline(0, color='black', lw=0.8, ls='--')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(coef_labels, fontsize=11)
ax3.set_xlabel('Coefficient (95% CI)')
ax3.set_title('Panel 3: ECM Coefficients  (red = p<0.05)', fontsize=11)
# 계수값 표시
for i, (c, e) in enumerate(zip(coefs, errors)):
    ax3.text(c + (e+0.001)*np.sign(c) + 0.002, i,
             f'{c:.4f}', va='center', ha='left' if c>0 else 'right', fontsize=9)

plt.tight_layout()
fig.savefig(
    "/sessions/gracious-tender-mccarthy/mnt/seoul house pricing/tokyo_analysis/figures/tokyo_ecm.png",
    dpi=150, bbox_inches='tight'
)
print("\n그래프 저장: tokyo_ecm.png")

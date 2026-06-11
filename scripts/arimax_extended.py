"""
ARIMAX(1,0,0) 확장 모형
- 기존 변수: mortgage_rate, csi, moon_dummy, yoon_dummy
- 추가 변수: jeonse_index (전세가격지수), mortgage_balance (주담대 잔액), jeonse_ratio (전세가율)
- 입력: data/merged_multi_extended.csv (fetch_new_vars.py 실행 후 생성됨)
- 출력: 콘솔 비교표 + figures/arimax_extended.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

import os
BASE = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing"
)
DATA_PATH   = os.path.join(BASE, "data", "merged_multi_extended.csv")
OUT_FIG     = os.path.join(BASE, "figures", "arimax_extended.png")


# ── 데이터 로드 ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")
df = df.sort_values("date").dropna(subset=["mom_change_pct", "mortgage_rate", "csi"])

y = df["mom_change_pct"]

# ── 모형별 변수 설정 ─────────────────────────────────────────────
MODELS = {}

# [M1] 기존 ARIMAX (베이스라인)
MODELS["M1 기존(베이스라인)"] = {
    "vars": ["mortgage_rate", "csi", "moon_dummy", "yoon_dummy"],
    "desc": "mortgage_rate, csi, moon_dummy, yoon_dummy"
}

# [M2] 전세가격지수 추가
if "jeonse_index" in df.columns and df["jeonse_index"].notna().sum() > 30:
    MODELS["M2 +전세가격지수"] = {
        "vars": ["mortgage_rate", "csi", "jeonse_index", "moon_dummy", "yoon_dummy"],
        "desc": "+ jeonse_index"
    }

# [M3] 주담대 잔액 추가
if "mortgage_balance" in df.columns and df["mortgage_balance"].notna().sum() > 30:
    MODELS["M3 +주담대잔액"] = {
        "vars": ["mortgage_rate", "csi", "mortgage_balance", "moon_dummy", "yoon_dummy"],
        "desc": "+ mortgage_balance"
    }

# [M4] 전세가율(갭투자 프록시) 추가
if "jeonse_ratio" in df.columns and df["jeonse_ratio"].notna().sum() > 30:
    MODELS["M4 +전세가율"] = {
        "vars": ["mortgage_rate", "csi", "jeonse_ratio", "moon_dummy", "yoon_dummy"],
        "desc": "+ jeonse_ratio(갭투자 프록시)"
    }

# [M5] 전체 추가 변수 (가능한 경우)
full_vars = ["mortgage_rate", "csi", "moon_dummy", "yoon_dummy"]
full_desc_parts = []
if "jeonse_index" in df.columns and df["jeonse_index"].notna().sum() > 30:
    full_vars.append("jeonse_index")
    full_desc_parts.append("jeonse_index")
if "mortgage_balance" in df.columns and df["mortgage_balance"].notna().sum() > 30:
    full_vars.append("mortgage_balance")
    full_desc_parts.append("mortgage_balance")
if len(full_desc_parts) >= 2:
    MODELS["M5 전체확장"] = {
        "vars": full_vars,
        "desc": "모든 추가 변수 포함"
    }


# ── 각 모형 추정 ────────────────────────────────────────────────
print("=" * 70)
print("ARIMAX(1,0,0) 확장 모형 비교")
print("=" * 70)

results_summary = {}

for name, cfg in MODELS.items():
    X_cols = cfg["vars"]
    # 해당 변수 컬럼이 모두 있는 행만 사용
    valid = df[X_cols + ["mom_change_pct"]].dropna()
    if len(valid) < 30:
        print(f"\n[{name}] 유효 데이터 부족 ({len(valid)}행) → 건너뜀")
        continue

    y_m = valid["mom_change_pct"]
    X_m = valid[X_cols]

    model = SARIMAX(endog=y_m, exog=X_m, order=(1, 0, 0), trend="c")
    res = model.fit(disp=False)
    dw = durbin_watson(res.resid)

    results_summary[name] = {
        "N": len(valid),
        "AIC": res.aic,
        "BIC": res.bic,
        "DW": dw,
        "AR1": res.params.get("ar.L1", float("nan")),
        "result_obj": res,
        "X_cols": X_cols,
        "y_used": y_m,
        "X_used": X_m,
    }

    print(f"\n{'─'*60}")
    print(f"[{name}]  N={len(valid)}  AIC={res.aic:.1f}  BIC={res.bic:.1f}  DW={dw:.3f}")
    print(f"{'─'*60}")
    # 계수 표 출력
    params = res.params
    pvals  = res.pvalues
    for var in params.index:
        stars = "***" if pvals[var] < 0.01 else "**" if pvals[var] < 0.05 else "*" if pvals[var] < 0.1 else ""
        print(f"  {var:<28} β={params[var]:>8.4f}   p={pvals[var]:.3f} {stars}")


# ── 요약 비교표 ─────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("모형 비교 요약")
print("=" * 70)
print(f"{'모형':<22} {'N':>5} {'AIC':>9} {'BIC':>9} {'DW':>7} {'AR(1)':>8}")
print("─" * 65)
for name, r in results_summary.items():
    print(f"{name:<22} {r['N']:>5} {r['AIC']:>9.1f} {r['BIC']:>9.1f} {r['DW']:>7.3f} {r['AR1']:>8.4f}")
print("\n* AIC/BIC가 낮을수록 좋음  * DW가 2에 가까울수록 자기상관 없음")


# ── 시각화: M1 vs 최고 모형 ─────────────────────────────────────
if len(results_summary) >= 2:
    model_names = list(results_summary.keys())
    m1 = results_summary[model_names[0]]
    best_name = min(results_summary, key=lambda k: results_summary[k]["AIC"])
    m_best = results_summary[best_name]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # 상단: 실제 vs M1 vs 최고 모형
    axes[0].plot(m1["y_used"].index, m1["y_used"].values,
                 label="Actual", color="#1a1a2e", linewidth=1.2)
    axes[0].plot(m1["result_obj"].fittedvalues.index,
                 m1["result_obj"].fittedvalues.values,
                 label="M1 기존", color="#e74c3c", linestyle="--", linewidth=1.1, alpha=0.8)
    if best_name != model_names[0]:
        axes[0].plot(m_best["result_obj"].fittedvalues.index,
                     m_best["result_obj"].fittedvalues.values,
                     label=f"{best_name}", color="#2ecc71", linestyle=":", linewidth=1.3)
    axes[0].set_title("ARIMAX 확장 모형: Actual vs Fitted")
    axes[0].set_ylabel("MoM Change (%)")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # 하단: AIC 비교 막대
    names = list(results_summary.keys())
    aics  = [results_summary[n]["AIC"] for n in names]
    colors = ["#3498db" if n == best_name else "#bdc3c7" for n in names]
    axes[1].barh(names, aics, color=colors)
    axes[1].set_xlabel("AIC (낮을수록 좋음)")
    axes[1].set_title("모형별 AIC 비교 (파란색 = 최고)")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {OUT_FIG}")
    plt.show()
else:
    print("\n[!] 비교 가능한 모형이 1개뿐 — 새 변수 데이터 파일을 먼저 준비하세요.")
    print("    fetch_new_vars.py 스크립트의 다운로드 안내를 따라주세요.")

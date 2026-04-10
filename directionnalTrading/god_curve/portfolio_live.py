"""
portfolio_live.py — Analyse combinée BTC + ETH (configs live actuelles)

Configs :
  BTC : vol_net_lin  sl=3.0   int=0  cap=0.70  net0.5h>60   max_orders=2
  ETH : vol_trend_lin sl=0.20  int=0  cap=0.55  tre2h>0.50   max_orders=2

Deux scénarios de sizing :
  A. Dynamique : scale BTC+ETH ensemble → maxDD_cumulé = $500
     (les deux assets sont scalés proportionnellement au même facteur)
  B. Fixe      : BTC $100/ordre × 2, ETH $150/ordre × 2

Sorties (dans portfolio_live/) :
  equity_A.png   : equity BTC+ETH+combined, sizing dynamique
  equity_B.png   : equity BTC+ETH+combined, sizing fixe
  dd_corr.png    : corrélation des drawdowns (overlay + scatter)
  stats.txt      : métriques + recommandation de répartition
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "portfolio_live"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from chart_utils import (
    load_contracts, precompute_vol, attach_cnet_data, run_config_tf,
    build_cumulative, build_hour_index, make_xlabels,
    TIMEFRAMES, VOL_LBS,
)

# ── Configs live ──────────────────────────────────────────────────────────────
CFG_BTC = dict(
    label='vrs_lin', curve='linear',
    slope=10.0, intercept=0.0, A_exp=None, tau=None,
    eq_cap=0.55, max_losses_cb=None,
    vol_lb_h=None, vol_thresh=None,
    vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
    vrs_g_min=0.5, vrs_g_max=1.5,
    max_orders=2,
)
CFG_ETH = dict(
    label='vol_trend_lin', curve='linear',
    slope=0.20, intercept=0.0, A_exp=None, tau=None,
    eq_cap=0.55, max_losses_cb=None,
    vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
    max_orders=1,
)

# Sizing fixe (scénario B)
BTC_SIZE_B = 100.0   # $ par ordre
ETH_SIZE_B = 150.0   # $ par ordre
MAX_DD_A   = 500.0   # $ cible drawdown combiné (scénario A)


# ── Chargement contrats ───────────────────────────────────────────────────────
def load_asset(csv_paths, label):
    print(f"\n  [{label}] Chargement...", flush=True)
    contracts_by_tf = []
    m5_ref = None
    for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
        try:
            cts = load_contracts(csv_paths, tf_floor, bid_up, bid_down, ask_up, ask_down)
            if tf_floor == '5min':
                precompute_vol(cts, VOL_LBS)
                m5_ref = cts
            else:
                precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
            contracts_by_tf.append(cts)
            print(f"      {tf_floor}: {len(cts)} contracts", flush=True)
        except Exception as e:
            print(f"      {tf_floor}: skipped ({e})", flush=True)
            contracts_by_tf.append([])
    if m5_ref:
        for cts in contracts_by_tf:
            if cts:
                attach_cnet_data(cts, (2, 5, 10), ref_contracts=m5_ref)
    return contracts_by_tf


def run_asset(contracts_by_tf, cfg, hour_index, n_hours, base_size):
    """Run config sur les 3 TF, retourne (cum, wins, losses, pnl_by_hour)."""
    cfg_sized = {**cfg, 'base_size_override': base_size}
    combined_pnl = defaultdict(float)
    wins = losses = 0
    for contracts in contracts_by_tf:
        ph, w, l = run_config_tf_sized(contracts, cfg, hour_index, base_size)
        for h, pnl in ph.items():
            combined_pnl[h] += pnl
        wins += w; losses += l
    cum, _ = build_cumulative(combined_pnl, n_hours)
    return cum, wins, losses, combined_pnl


def run_config_tf_sized(contracts, cfg, hour_index, base_size):
    """Wrapper run_config_tf avec BASE_SIZE personnalisé."""
    import chart_utils as cu
    orig = cu.BASE_SIZE
    cu.BASE_SIZE = float(base_size)
    try:
        result = run_config_tf(contracts, cfg, hour_index)
    finally:
        cu.BASE_SIZE = orig
    return result


# ── Stats ─────────────────────────────────────────────────────────────────────
def stats(cum, wins, losses, days, label, base_size, max_orders):
    total  = wins + losses
    wr     = wins / total * 100 if total else 0
    dd     = float((np.maximum.accumulate(cum) - cum).max())
    pnl_j  = cum[-1] / days if days > 0 else 0
    rf     = cum[-1] / dd if dd > 1e-9 else float('inf')
    return dict(label=label, total=total, wr=wr, pnl_total=cum[-1],
                pnl_j=pnl_j, maxdd=dd, rf=rf,
                base_size=base_size, max_orders=max_orders, days=days)


def print_stats(s, f=None):
    rf_str = f"{s['rf']:.1f}x" if s['rf'] != float('inf') else "inf"
    lines = [
        f"  {s['label']}",
        f"    Trades    : {s['total']}  WR={s['wr']:.1f}%",
        f"    PnL total : ${s['pnl_total']:,.0f}  (${s['pnl_j']:.0f}/j)",
        f"    MaxDD     : ${s['maxdd']:.0f}",
        f"    RF (PnL_total / maxDD) : {rf_str}",
        f"    Sizing    : ${s['base_size']:.0f}/ordre × {s['max_orders']} ordres",
    ]
    for l in lines:
        print(l, flush=True)
        if f: f.write(l + "\n")


# ── Graphique equity ──────────────────────────────────────────────────────────
def plot_equity(cum_btc, cum_eth, cum_comb, hour_index, n_hours, days,
                title, out_path, label_a="Sizing dynamique"):
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)

    dd_btc  = np.maximum.accumulate(cum_btc)  - cum_btc
    dd_eth  = np.maximum.accumulate(cum_eth)  - cum_eth
    dd_comb = np.maximum.accumulate(cum_comb) - cum_comb

    wr_hint = ""
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.plot(x, cum_btc,  color='#FFC107', linewidth=1.4, label=f'BTC  ${cum_btc[-1]:,.0f}  (${cum_btc[-1]/days:.0f}/j)')
    ax1.plot(x, cum_eth,  color='#4CAF50', linewidth=1.4, label=f'ETH  ${cum_eth[-1]:,.0f}  (${cum_eth[-1]/days:.0f}/j)')
    ax1.plot(x, cum_comb, color='#2196F3', linewidth=2.0, label=f'BTC+ETH  ${cum_comb[-1]:,.0f}  (${cum_comb[-1]/days:.0f}/j)')
    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_title(f"{title}  ({days:.0f}j)", fontsize=12)
    ax1.set_ylabel("PnL cumulé ($)")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30)

    ax2.plot(x, dd_btc,  color='#FFC107', linewidth=1.0, alpha=0.8, label='DD BTC')
    ax2.plot(x, dd_eth,  color='#4CAF50', linewidth=1.0, alpha=0.8, label='DD ETH')
    ax2.plot(x, dd_comb, color='#2196F3', linewidth=1.6,             label='DD combiné')
    ax2.invert_yaxis()
    ax2.set_ylabel("Drawdown ($)")
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → {out_path}", flush=True)


# ── Graphique corrélation DD ──────────────────────────────────────────────────
def plot_dd_corr(cum_btc, cum_eth, cum_comb, hour_index, n_hours, out_path):
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)

    dd_btc  = np.maximum.accumulate(cum_btc)  - cum_btc
    dd_eth  = np.maximum.accumulate(cum_eth)  - cum_eth
    dd_comb = np.maximum.accumulate(cum_comb) - cum_comb

    # Corrélation sur heures où les deux sont en DD
    both_dd = (dd_btc > 0) & (dd_eth > 0)
    corr = float(np.corrcoef(dd_btc, dd_eth)[0, 1])
    corr_when_both = float(np.corrcoef(dd_btc[both_dd], dd_eth[both_dd])[0, 1]) if both_dd.sum() > 10 else 0.0

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1 : DD overlay ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(x, dd_btc,  alpha=0.3, color='#FFC107', label='DD BTC')
    ax1.fill_between(x, dd_eth,  alpha=0.3, color='#4CAF50', label='DD ETH')
    ax1.plot(x, dd_comb, color='#2196F3', linewidth=1.6, label='DD combiné')

    # Zones où les deux sont simultanément en DD
    simultaneous = (dd_btc > 5) & (dd_eth > 5)
    if simultaneous.any():
        ax1.fill_between(x, 0, np.maximum(dd_btc, dd_eth),
                         where=simultaneous, alpha=0.25, color='red',
                         label=f'DD simultanés ({simultaneous.sum()}h)')

    ax1.invert_yaxis()
    ax1.set_title(f"Drawdowns BTC vs ETH  (corr={corr:.2f}, corr quand les deux en DD={corr_when_both:.2f})")
    ax1.set_ylabel("Drawdown ($)")
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30)

    # ── Panel 2 : Scatter DD_btc vs DD_eth ──
    ax2 = fig.add_subplot(gs[1, 0])
    sc = ax2.scatter(dd_btc, dd_eth, c=x, cmap='viridis', s=3, alpha=0.4)
    plt.colorbar(sc, ax=ax2, label='heure')
    ax2.set_xlabel("DD BTC ($)")
    ax2.set_ylabel("DD ETH ($)")
    ax2.set_title(f"Scatter DD — corr={corr:.2f}")
    ax2.grid(alpha=0.3)

    # ── Panel 3 : % heures en DD simultané par tranche horaire ──
    ax3 = fig.add_subplot(gs[1, 1])
    # Histogram du % simultané par jour
    n_days = max(1, n_hours // 24)
    day_pct = []
    for d in range(n_days):
        s_d = simultaneous[d*24 : (d+1)*24]
        day_pct.append(100 * s_d.sum() / max(len(s_d), 1))
    ax3.bar(range(len(day_pct)), day_pct, color='red', alpha=0.6)
    ax3.axhline(np.mean(day_pct), color='darkred', linestyle='--',
                label=f'moy={np.mean(day_pct):.0f}%')
    ax3.set_xlabel("Jour")
    ax3.set_ylabel("% heures DD simultané")
    ax3.set_title("DD simultané (DD_btc>5$ ET DD_eth>5$) par jour")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → {out_path}", flush=True)


# ── Recommandation sizing ─────────────────────────────────────────────────────
def recommend(s_btc, s_eth, cum_btc_unit, cum_eth_unit, n_hours, f=None):
    """
    Calcule la répartition optimale BTC/ETH.
    Entrée : cums calculés avec BASE_SIZE=1$ (unit).
    """
    hourly_btc = np.diff(np.concatenate([[0.0], cum_btc_unit]))
    hourly_eth = np.diff(np.concatenate([[0.0], cum_eth_unit]))

    mu_btc  = np.mean(hourly_btc);    mu_eth  = np.mean(hourly_eth)
    sig_btc = np.std(hourly_btc);     sig_eth = np.std(hourly_eth)
    corr_pnl = float(np.corrcoef(hourly_btc, hourly_eth)[0, 1])

    # Sharpe horaire (per-$)
    sr_btc = mu_btc / sig_btc if sig_btc > 0 else 0
    sr_eth = mu_eth / sig_eth if sig_eth > 0 else 0

    # RF = PnL_total / maxDD (per-$)
    rf_btc = s_btc['rf']
    rf_eth = s_eth['rf']

    # Allocation min-variance (2 actifs)
    # w* = (sig_eth^2 - cov) / (sig_btc^2 + sig_eth^2 - 2*cov)
    cov = corr_pnl * sig_btc * sig_eth
    denom = sig_btc**2 + sig_eth**2 - 2 * cov
    if abs(denom) > 1e-12:
        w_btc_mv = (sig_eth**2 - cov) / denom
        w_btc_mv = max(0.1, min(0.9, w_btc_mv))
    else:
        w_btc_mv = 0.5
    w_eth_mv = 1.0 - w_btc_mv

    # Allocation max-Sharpe (approx proportionnel au SR)
    sr_tot = abs(sr_btc) + abs(sr_eth) if (abs(sr_btc) + abs(sr_eth)) > 0 else 1
    w_btc_sr = abs(sr_btc) / sr_tot
    w_eth_sr = abs(sr_eth) / sr_tot

    lines = [
        "",
        "=== RECOMMANDATION SIZING ===",
        f"  Corrélation PnL horaire BTC↔ETH : {corr_pnl:.3f}",
        f"  Sharpe horaire BTC : {sr_btc:.4f}  ETH : {sr_eth:.4f}",
        f"  RF BTC : {rf_btc:.1f}x  RF ETH : {rf_eth:.1f}x",
        f"  Ratio Sharpe BTC/ETH : {sr_btc/sr_eth:.2f}" if sr_eth != 0 else "",
        "",
        f"  Allocation min-variance  : BTC {w_btc_mv*100:.0f}%  /  ETH {w_eth_mv*100:.0f}%",
        f"  Allocation max-Sharpe    : BTC {w_btc_sr*100:.0f}%  /  ETH {w_eth_sr*100:.0f}%",
        "",
        "  Interprétation :",
    ]
    if corr_pnl < 0.15:
        lines.append("  Les deux stratégies sont peu corrélées → diversification efficace.")
        lines.append("  Répartition optimale proche de min-variance.")
    elif corr_pnl < 0.40:
        lines.append("  Corrélation modérée → diversification partielle.")
        lines.append("  Utiliser un mix min-variance / max-Sharpe.")
    else:
        lines.append("  Corrélation élevée → peu de bénéfice à diversifier.")
        lines.append("  Concentrer sur l'asset avec le meilleur Sharpe.")

    rec_btc = 0.5 * (w_btc_mv + w_btc_sr)
    rec_eth = 1 - rec_btc
    lines += [
        "",
        f"  Répartition recommandée : BTC {rec_btc*100:.0f}%  /  ETH {rec_eth*100:.0f}%",
        f"  Exemple pour $500 cible exposure : BTC ${rec_btc*500:.0f}  ETH ${rec_eth*500:.0f}",
        "",
        "  Note sizing fixe actuel : BTC $100×2=$200  ETH $150×1=$150 → 57% / 43%",
        f"  vs recommandé           : BTC {rec_btc*100:.0f}%  ETH {rec_eth*100:.0f}%",
    ]
    for l in lines:
        print(l, flush=True)
        if f: f.write(l + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_btc = [str(BASE / "Datas" / "csv" / "BTC.csv")]
    csv_eth = [str(BASE / "Datas" / "csv" / "ETH.csv")]

    print("=== Chargement BTC ===")
    ctf_btc = load_asset(csv_btc, "BTC")
    print("=== Chargement ETH ===")
    ctf_eth = load_asset(csv_eth, "ETH")

    # Index horaire commun (union BTC ∪ ETH)
    all_ctf = ctf_btc + ctf_eth
    hour_index, n_hours = build_hour_index(all_ctf)
    days = n_hours / 24
    print(f"\n  Timeline commune : {n_hours} heures ({days:.1f}j)", flush=True)

    # ── Scénario A : sizing dynamique ────────────────────────────────────────
    print("\n=== Scénario A : sizing dynamique (maxDD combiné = $500) ===")
    cum_btc_u, w_btc, l_btc, _ = run_asset(ctf_btc, CFG_BTC, hour_index, n_hours, base_size=1.0)
    cum_eth_u, w_eth, l_eth, _ = run_asset(ctf_eth, CFG_ETH, hour_index, n_hours, base_size=1.0)
    cum_comb_u = cum_btc_u + cum_eth_u
    maxdd_u = float((np.maximum.accumulate(cum_comb_u) - cum_comb_u).max())
    scale_a = MAX_DD_A / maxdd_u if maxdd_u > 1 else 1.0
    print(f"  maxDD unit = ${maxdd_u:.4f} → scale = {scale_a:.2f}x", flush=True)

    cum_btc_a  = cum_btc_u  * scale_a
    cum_eth_a  = cum_eth_u  * scale_a
    cum_comb_a = cum_comb_u * scale_a

    eff_size_btc_a = scale_a  # $ effectif par ordre = 1$ × scale
    eff_size_eth_a = scale_a

    s_btc_a = stats(cum_btc_a,  w_btc, l_btc, days, "BTC (A)", eff_size_btc_a, CFG_BTC['max_orders'])
    s_eth_a = stats(cum_eth_a,  w_eth, l_eth, days, "ETH (A)", eff_size_eth_a, CFG_ETH['max_orders'])
    s_cmb_a = stats(cum_comb_a, w_btc+w_eth, l_btc+l_eth, days, "BTC+ETH (A)", scale_a, 2)

    plot_equity(cum_btc_a, cum_eth_a, cum_comb_a, hour_index, n_hours, days,
                f"Portfolio live — sizing dynamique (maxDD→$500, scale={scale_a:.1f}x)",
                OUT_DIR / "equity_A.png")

    # ── Scénario B : sizing fixe ──────────────────────────────────────────────
    print("\n=== Scénario B : sizing fixe (BTC $100×2, ETH $150×1) ===")
    cum_btc_b, _, _, _ = run_asset(ctf_btc, CFG_BTC, hour_index, n_hours, base_size=BTC_SIZE_B)
    cum_eth_b, _, _, _ = run_asset(ctf_eth, CFG_ETH, hour_index, n_hours, base_size=ETH_SIZE_B)
    cum_comb_b = cum_btc_b + cum_eth_b

    s_btc_b = stats(cum_btc_b,  w_btc, l_btc, days, "BTC (B)", BTC_SIZE_B, CFG_BTC['max_orders'])
    s_eth_b = stats(cum_eth_b,  w_eth, l_eth, days, "ETH (B)", ETH_SIZE_B, CFG_ETH['max_orders'])
    s_cmb_b = stats(cum_comb_b, w_btc+w_eth, l_btc+l_eth, days, "BTC+ETH (B)", 0, 0)

    plot_equity(cum_btc_b, cum_eth_b, cum_comb_b, hour_index, n_hours, days,
                f"Portfolio live — sizing fixe (BTC $100×2 / ETH $150×1)",
                OUT_DIR / "equity_B.png")

    # ── Corrélation DD ────────────────────────────────────────────────────────
    # On utilise le scénario A pour la corrélation (scale identique)
    plot_dd_corr(cum_btc_a, cum_eth_a, cum_comb_a, hour_index, n_hours,
                 OUT_DIR / "dd_corr.png")

    # ── Scénario C : sizing recommandé (32% BTC / 68% ETH, même budget $500) ──
    # Budget total = $500 exposure, réparti 32/68, max_orders=2
    # size/ordre = budget_asset / max_orders
    BUDGET_C      = 500.0
    REC_BTC_PCT   = 0.32
    REC_ETH_PCT   = 0.68
    btc_size_c    = (BUDGET_C * REC_BTC_PCT) / CFG_BTC['max_orders']  # $/ordre
    eth_size_c    = (BUDGET_C * REC_ETH_PCT) / CFG_ETH['max_orders']
    cum_btc_c     = cum_btc_u * btc_size_c
    cum_eth_c     = cum_eth_u * eth_size_c
    cum_comb_c    = cum_btc_c + cum_eth_c
    s_btc_c = stats(cum_btc_c,  w_btc, l_btc, days, "BTC (C)", btc_size_c, CFG_BTC['max_orders'])
    s_eth_c = stats(cum_eth_c,  w_eth, l_eth, days, "ETH (C)", eth_size_c, CFG_ETH['max_orders'])
    s_cmb_c = stats(cum_comb_c, w_btc+w_eth, l_btc+l_eth, days, "BTC+ETH (C)", 0, 0)

    plot_equity(cum_btc_c, cum_eth_c, cum_comb_c, hour_index, n_hours, days,
                f"Portfolio live — sizing recommandé (BTC ${btc_size_c:.0f}×2 / ETH ${eth_size_c:.0f}×2)",
                OUT_DIR / "equity_C.png")

    # ── Stats + Recommandation ────────────────────────────────────────────────
    with open(OUT_DIR / "stats.txt", "w", encoding="utf-8") as f:
        f.write("=== PORTFOLIO LIVE BTC + ETH ===\n\n")

        f.write("── Scénario A : sizing dynamique (maxDD combiné = $500) ──\n")
        for s in [s_btc_a, s_eth_a, s_cmb_a]:
            print_stats(s, f); f.write("\n")

        f.write("\n── Scénario B : sizing fixe ACTUEL (BTC $100×2, ETH $150×1) ──\n")
        for s in [s_btc_b, s_eth_b, s_cmb_b]:
            print_stats(s, f); f.write("\n")

        f.write(f"\n── Scénario C : sizing recommandé (BTC ${btc_size_c:.0f}×2, ETH ${eth_size_c:.0f}×2) ──\n")
        for s in [s_btc_c, s_eth_c, s_cmb_c]:
            print_stats(s, f); f.write("\n")

        recommend(s_btc_a, s_eth_a, cum_btc_u, cum_eth_u, n_hours, f)

    # ── Tableau comparatif B vs C ─────────────────────────────────────────────
    print("\n=== COMPARATIF : actuel (B) vs recommandé (C) ===")
    rows = [
        ("",              "BTC",          "ETH",          "COMBINÉ"),
        ("── Actuel (B)", f"${s_btc_b['pnl_j']:.0f}/j  maxDD=${s_btc_b['maxdd']:.0f}  RF={s_btc_b['rf']:.0f}x",
                          f"${s_eth_b['pnl_j']:.0f}/j  maxDD=${s_eth_b['maxdd']:.0f}  RF={s_eth_b['rf']:.0f}x",
                          f"${s_cmb_b['pnl_j']:.0f}/j  maxDD=${s_cmb_b['maxdd']:.0f}  RF={s_cmb_b['rf']:.0f}x"),
        (f"── Recommandé (C)\n   BTC ${btc_size_c:.0f}×2 / ETH ${eth_size_c:.0f}×2",
                          f"${s_btc_c['pnl_j']:.0f}/j  maxDD=${s_btc_c['maxdd']:.0f}  RF={s_btc_c['rf']:.0f}x",
                          f"${s_eth_c['pnl_j']:.0f}/j  maxDD=${s_eth_c['maxdd']:.0f}  RF={s_eth_c['rf']:.0f}x",
                          f"${s_cmb_c['pnl_j']:.0f}/j  maxDD=${s_cmb_c['maxdd']:.0f}  RF={s_cmb_c['rf']:.0f}x"),
        ("── Δ (C - B)",  f"Δ PnL/j ${s_btc_c['pnl_j']-s_btc_b['pnl_j']:+.0f}",
                          f"Δ PnL/j ${s_eth_c['pnl_j']-s_eth_b['pnl_j']:+.0f}",
                          f"Δ PnL/j ${s_cmb_c['pnl_j']-s_cmb_b['pnl_j']:+.0f}  Δ maxDD ${s_cmb_c['maxdd']-s_cmb_b['maxdd']:+.0f}"),
    ]
    for row in rows:
        print(f"  {row[0]:<35} BTC={row[1]:<40} ETH={row[2]:<40} COMB={row[3]}", flush=True)

    print(f"\n  Stats -> {OUT_DIR / 'stats.txt'}")
    print(f"  equity_A/B/C.png, dd_corr.png → {OUT_DIR}")

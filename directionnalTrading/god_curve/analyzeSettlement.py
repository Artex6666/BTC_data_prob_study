#!/usr/bin/env python3
"""
analyzeSettlement.py — Compare les outcomes ask-based (BT) vs Gamma (réel).

Charge settlement.csv (généré par scrapSettlement.py) et les CSV de prix,
puis compare pour chaque contrat si la direction prédite par le BT correspond
à la vraie résolution Gamma.

Outputs :
  - Stats globales (N wrong, % wrong, par asset/TF/période)
  - Chart : % wrong outcomes par jour (timeline)
  - settlement_analysis.csv : contrats mal résolus par le BT

Usage:
    python analyzeSettlement.py [--no-chart]
"""

import sys, argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE_DIR     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
SETTLEMENT   = Path(__file__).resolve().parent.parent / "Datas" / "csv" / "settlement.csv"
OUT_ANALYSIS = Path(__file__).resolve().parent.parent / "Datas" / "csv" / "settlement_analysis.csv"

ASSETS = {
    'btc': 'BTC.csv',
    'eth': 'ETH.csv',
}
TF_COLS = {
    '5min':  ('m5_up_ask',  'm5_down_ask'),
    '15min': ('m15_up_ask', 'm15_down_ask'),
    '1h':    ('h1_up_ask',  'h1_down_ask'),
}
TF_DUR = {'5min': 300, '15min': 900, '1h': 3600}


# ── Chargement du settlement ────────────────────────────────────────────────────
def load_settlement() -> pd.DataFrame:
    df = pd.read_csv(SETTLEMENT)
    # Exclure les non trouvés (outcome_up vide)
    df = df[df['outcome_up'].notna() & (df['outcome_up'] != '')]
    df['outcome_up']   = df['outcome_up'].astype(float)
    df['outcome_down'] = df['outcome_down'].astype(float)
    df['open_ts'] = df['open_ts'].astype(int)
    df['ce_ts']   = df['ce_ts'].astype(int)
    # Gamma winner : outcome_up = 1.0 → UP won; outcome_down = 1.0 → DOWN won
    df['gamma_up_won']   = df['outcome_up']   > 0.5
    df['gamma_down_won'] = df['outcome_down'] > 0.5
    df['gamma_dir'] = df['gamma_up_won'].map({True: 'UP', False: 'DOWN'})
    return df


# ── Chargement des ask finaux depuis le CSV prix ────────────────────────────────
def load_ask_finals(asset: str, tf: str) -> dict[int, str]:
    """
    Retourne open_ts → 'UP'|'DOWN' basé sur up_ask[-1] > down_ask[-1]
    (logique BT actuelle).
    """
    csv_path = BASE_DIR / ASSETS[asset]
    if not csv_path.exists():
        return {}

    ask_up_col, ask_down_col = TF_COLS[tf]
    try:
        df = pd.read_csv(csv_path, usecols=['timestamp', ask_up_col, ask_down_col])
    except ValueError:
        return {}

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    offset = pd.tseries.frequencies.to_offset(tf)
    df['contract'] = df['timestamp'].dt.floor(tf) + offset

    result = {}
    for ce_dt, grp in df.groupby('contract'):
        ce_dt   = pd.Timestamp(ce_dt)
        open_dt = ce_dt - offset
        open_ts = int(open_dt.timestamp())

        # Dernier tick avant expiry (colonne up_ask finale)
        last = grp.iloc[-1]
        ua   = float(last[ask_up_col])
        da   = float(last[ask_down_col])

        bt_dir = 'UP' if ua > da else 'DOWN'
        result[open_ts] = bt_dir

    return result


# ── Analyse principale ─────────────────────────────────────────────────────────
def run(show_chart: bool):
    if not SETTLEMENT.exists():
        print(f"settlement.csv introuvable : {SETTLEMENT}")
        print("Lance d'abord scrapSettlement.py")
        sys.exit(1)

    print("Chargement settlement.csv...")
    sdf = load_settlement()
    print(f"  {len(sdf)} contrats avec outcome Gamma\n")

    rows = []

    for asset in ASSETS:
        for tf in ['5min', '15min', '1h']:
            subset = sdf[(sdf['asset'] == asset) & (sdf['tf'] == tf)].copy()
            if subset.empty:
                continue

            print(f"Chargement ask finaux {asset.upper()} {tf}...", end=' ', flush=True)
            bt_finals = load_ask_finals(asset, tf)
            print(f"{len(bt_finals)} contrats BT")

            for _, row in subset.iterrows():
                open_ts  = int(row['open_ts'])
                bt_dir   = bt_finals.get(open_ts)
                gamma_dir = row['gamma_dir']

                if bt_dir is None:
                    continue

                is_wrong = bt_dir != gamma_dir
                rows.append({
                    'slug':       row['slug'],
                    'asset':      asset,
                    'tf':         tf,
                    'open_ts':    open_ts,
                    'ce_ts':      int(row['ce_ts']),
                    'bt_dir':     bt_dir,
                    'gamma_dir':  gamma_dir,
                    'outcome_up': row['outcome_up'],
                    'outcome_down': row['outcome_down'],
                    'wrong':      is_wrong,
                })

    adf = pd.DataFrame(rows)
    if adf.empty:
        print("Aucune donnée croisée. Vérifie les fichiers CSV et settlement.csv")
        sys.exit(1)

    adf['date'] = pd.to_datetime(adf['open_ts'], unit='s', utc=True).dt.date

    # ── Stats globales ─────────────────────────────────────────────────────────
    total   = len(adf)
    wrong   = adf['wrong'].sum()
    pct_wr  = 100 * wrong / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"STATS GLOBALES")
    print(f"  Total contrats analysés : {total}")
    print(f"  Directions incorrectes  : {wrong}  ({pct_wr:.2f}%)")
    print(f"  Directions correctes    : {total - wrong}  ({100 - pct_wr:.2f}%)")

    # ── Par asset ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PAR ASSET")
    for asset in adf['asset'].unique():
        sub = adf[adf['asset'] == asset]
        w   = sub['wrong'].sum()
        t   = len(sub)
        print(f"  {asset.upper():<6}  {t:4d} contrats  {w:3d} wrong ({100*w/t:.1f}%)")

    # ── Par TF ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PAR TIMEFRAME")
    for tf in ['5min', '15min', '1h']:
        sub = adf[adf['tf'] == tf]
        if sub.empty: continue
        w = sub['wrong'].sum()
        t = len(sub)
        print(f"  {tf:<8}  {t:4d} contrats  {w:3d} wrong ({100*w/t:.1f}%)")

    # ── Par jour ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PAR JOUR (wrong outcomes %)")
    daily = adf.groupby('date').agg(
        total=('wrong', 'count'),
        wrong=('wrong', 'sum'),
    ).reset_index()
    daily['pct_wrong'] = 100 * daily['wrong'] / daily['total']
    for _, dr in daily.iterrows():
        bar = '█' * int(dr['pct_wrong'] / 2)
        print(f"  {dr['date']}  {dr['total']:3d} ctrs  {dr['wrong']:3d} wrong  "
              f"{dr['pct_wrong']:5.1f}%  {bar}")

    # ── Par asset × TF ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"DETAIL ASSET × TF")
    pivot = adf.groupby(['asset', 'tf']).agg(
        total=('wrong', 'count'),
        wrong=('wrong', 'sum'),
    ).reset_index()
    pivot['pct_wrong'] = 100 * pivot['wrong'] / pivot['total']
    print(f"  {'Asset':<6} {'TF':<8} {'Total':>6} {'Wrong':>6} {'%Wrong':>7}")
    print(f"  {'-'*40}")
    for _, r in pivot.iterrows():
        print(f"  {r['asset'].upper():<6} {r['tf']:<8} {r['total']:6d} {r['wrong']:6d} {r['pct_wrong']:7.1f}%")

    # ── Exemples de wrong ─────────────────────────────────────────────────────
    wrong_df = adf[adf['wrong']].sort_values('open_ts')
    if len(wrong_df) > 0:
        print(f"\n{'─'*60}")
        print(f"EXEMPLES DE MAUVAIS OUTCOMES BT (premiers 20)")
        print(f"  {'Date (UTC)':<20} {'Asset':<5} {'TF':<7} {'BT':>4} {'Gamma':>6}  {'Slug'}")
        print(f"  {'-'*75}")
        for _, r in wrong_df.head(20).iterrows():
            dt_str = datetime.fromtimestamp(r['open_ts'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            print(f"  {dt_str:<20} {r['asset'].upper():<5} {r['tf']:<7} {r['bt_dir']:>4} {r['gamma_dir']:>6}  {r['slug']}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    adf.to_csv(OUT_ANALYSIS, index=False)
    print(f"\nAnalyse exportée : {OUT_ANALYSIS}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    if show_chart:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # Chart 1 : % wrong par jour global
            ax1 = axes[0]
            ax1.bar(daily['date'].astype(str), daily['pct_wrong'],
                    color='#E53935', alpha=0.8, label='% Wrong global')
            ax1.axhline(y=pct_wr, color='black', linestyle='--', linewidth=1,
                        label=f'Moyenne {pct_wr:.1f}%')
            ax1.set_title('Wrong Outcomes BT vs Gamma — par jour', fontsize=14)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('% Wrong')
            ax1.set_ylim(0, max(30, daily['pct_wrong'].max() + 5))
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

            # Chart 2 : % wrong par jour par asset (BTC vs ETH)
            ax2 = axes[1]
            colors = {'btc': '#F7931A', 'eth': '#627EEA'}
            for asset in adf['asset'].unique():
                sub = adf[adf['asset'] == asset]
                d = sub.groupby('date').agg(
                    total=('wrong', 'count'), wrong=('wrong', 'sum')
                ).reset_index()
                d['pct_wrong'] = 100 * d['wrong'] / d['total']
                ax2.plot(d['date'].astype(str), d['pct_wrong'],
                         marker='o', label=f'{asset.upper()}',
                         color=colors.get(asset, '#666'), linewidth=2)

            ax2.set_title('Wrong Outcomes par asset — par jour', fontsize=14)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('% Wrong')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            chart_path = Path(__file__).resolve().parent.parent / "Datas" / "settlement_wrong_outcomes.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Chart sauvegardé : {chart_path}")

        except ImportError:
            print("[SKIP] matplotlib non disponible")

    print(f"\nDone.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Analyse wrong outcomes BT vs Gamma')
    ap.add_argument('--no-chart', action='store_true', help='Ne génère pas le chart matplotlib')
    args = ap.parse_args()
    run(show_chart=not args.no_chart)

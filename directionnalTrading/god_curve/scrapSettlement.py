#!/usr/bin/env python3
"""
scrapSettlement.py — Scrape les vrais outcomes des contrats BTC/ETH depuis Gamma API.

Endpoint : https://gamma-api.polymarket.com/events/slug/<slug>
Slugs m5/m15 : btc-updown-5m-{open_ts}, btc-updown-15m-{open_ts}
               eth-updown-5m-{open_ts}, eth-updown-15m-{open_ts}
Slugs h1     : bitcoin-up-or-down-{month}-{day}-{year}-{hour}{am/pm}-et
               ethereum-up-or-down-{month}-{day}-{year}-{hour}{am/pm}-et

Output CSV : settlement.csv
  slug, asset, tf, open_ts, ce_ts, outcome_up, outcome_down

Usage:
    python scrapSettlement.py [--rate <req/s>] [--reset]
"""

import sys, time, json, argparse, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import deque
import threading

import pandas as pd
import numpy as np
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
OUT_CSV   = ROOT / "Datas" / "csv" / "settlement.csv"
API_BASE  = "https://gamma-api.polymarket.com/events/slug"

# Toutes les sources CSV par asset : on merge et déduplique les timestamps
# Format : { asset: [path1, path2, ...] }
ASSET_CSV_SOURCES = {
    'btc': [
        ROOT / "Datas"       / "csv"      / "BTC.csv",
        ROOT / "reportLive"  / "safeChase" / "BTC.csv",
    ],
    'eth': [
        ROOT / "Datas"       / "csv"      / "ETH.csv",
        ROOT / "reportLive"  / "safeChase" / "ETH.csv",
    ],
    'sol': [
        ROOT / "Datas"       / "csv"      / "SOL.csv",
        ROOT / "reportLive"  / "safeChase" / "SOL.csv",
    ],
    'bnb': [
        ROOT / "reportLive"  / "safeChase" / "BNB.csv",
    ],
    'xrp': [
        ROOT / "reportLive"  / "safeChase" / "XRP.csv",
    ],
}

TFS    = ['5min', '15min', '1h']
TF_DUR = {'5min': 300, '15min': 900, '1h': 3600}

# Slugs m5/m15 : {asset}-updown-{tf}-{open_ts}
TF_TO_SLUG = {'5min': '5m', '15min': '15m'}

# Slugs h1 : {crypto_full}-up-or-down-{month}-{day}-{year}-{hour}{am/pm}-et
CRYPTO_FULL = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'sol': 'solana',
    'bnb': 'bnb',
    'xrp': 'xrp',
}
MONTHS = ['january','february','march','april','may','june',
          'july','august','september','october','november','december']

_ET_TZ = ZoneInfo("America/New_York")  # gère automatiquement EST (UTC-5) et EDT (UTC-4)

CSV_FIELDS = ['slug', 'asset', 'tf', 'open_ts', 'ce_ts', 'outcome_up', 'outcome_down']


def make_h1_slug(asset: str, open_ts: int) -> str:
    """
    Genere le slug h1 format lisible.
    Avant le 2026-03-15 : bitcoin-up-or-down-february-14-6pm-et  (sans annee)
    A partir du 2026-03-15 : bitcoin-up-or-down-march-15-2026-12pm-et (avec annee)
    """
    dt_utc = datetime.fromtimestamp(open_ts, tz=timezone.utc)
    dt_et  = dt_utc.astimezone(_ET_TZ)
    month  = MONTHS[dt_et.month - 1]
    day    = dt_et.day
    year   = dt_et.year
    h      = dt_et.hour
    if h == 0:
        hour_str = '12am'
    elif h < 12:
        hour_str = f'{h}am'
    elif h == 12:
        hour_str = '12pm'
    else:
        hour_str = f'{h - 12}pm'
    crypto = CRYPTO_FULL.get(asset, asset)
    # Le 15 mars 2026, Polymarket a commence a inclure l'annee dans le slug
    YEAR_CUTOFF = 1773547200  # 2026-03-15 04:00 UTC = minuit EDT
    if open_ts >= YEAR_CUTOFF:
        return f'{crypto}-up-or-down-{month}-{day}-{year}-{hour_str}-et'
    else:
        return f'{crypto}-up-or-down-{month}-{day}-{hour_str}-et'


# ── Génération des slugs ────────────────────────────────────────────────────────
def gen_contracts_for_asset(asset: str) -> list[dict]:
    """
    Fusionne tous les CSV sources de l'asset, déduplique les timestamps,
    et retourne uniquement les contrats qui ont une couverture réelle dans le CSV
    (>= 2 ticks dans les 60 dernières secondes avant expiry, comme load_contracts).
    """
    csv_paths = [p for p in ASSET_CSV_SOURCES.get(asset, []) if p.exists()]
    if not csv_paths:
        return []

    dfs = []
    for p in csv_paths:
        try:
            d = pd.read_csv(p, usecols=['timestamp'])
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            dfs.append(d)
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")

    if not dfs:
        return []

    df = pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

    contracts = []
    seen: set[tuple] = set()

    for tf in TFS:
        offset    = pd.tseries.frequencies.to_offset(tf)
        td_60s    = pd.Timedelta(seconds=60)
        df['_ce'] = df['timestamp'].dt.floor(tf) + offset

        for ce_dt, grp in df.groupby('_ce'):
            ce_dt = pd.Timestamp(ce_dt)
            # Filtre : >= 2 ticks dans les 60 dernières secondes (même critère que load_contracts)
            t60 = grp[grp['timestamp'] >= ce_dt - td_60s]
            if len(t60) < 2:
                continue
            open_dt = ce_dt - offset
            open_ts = int(open_dt.timestamp())
            ce_ts   = int(ce_dt.timestamp())
            if tf == '1h':
                slug = make_h1_slug(asset, open_ts)
            else:
                slug = f"{asset}-updown-{TF_TO_SLUG[tf]}-{open_ts}"
            key = (slug, tf)
            if key in seen:
                continue
            seen.add(key)
            contracts.append({
                'slug':    slug,
                'asset':   asset,
                'tf':      tf,
                'open_ts': open_ts,
                'ce_ts':   ce_ts,
            })

        df.drop(columns=['_ce'], inplace=True)

    contracts.sort(key=lambda c: c['open_ts'])
    return contracts


# ── Chargement du CSV existant ─────────────────────────────────────────────────
def load_existing(out_csv: Path) -> set[str]:
    """Retourne l'ensemble des slugs déjà scrapés."""
    if not out_csv.exists():
        return set()
    done = set()
    with open(out_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row['slug'])
    return done


# ── Rate limiter ────────────────────────────────────────────────────────────────
class RateLimiter:
    """Token bucket thread-safe."""
    def __init__(self, rate: float):
        self.rate   = rate        # req/s
        self.tokens = rate
        self.last   = time.monotonic()
        self._lock  = threading.Lock()

    def acquire(self):
        with self._lock:
            now   = time.monotonic()
            delta = now - self.last
            self.last = now
            self.tokens = min(self.rate, self.tokens + delta * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
        sleep_t = (1.0 - self.tokens) / self.rate
        time.sleep(sleep_t)
        with self._lock:
            self.tokens = 0.0


# ── Fetch d'un contrat ─────────────────────────────────────────────────────────
def fetch_outcome(slug: str, timeout: int = 10) -> dict | None:
    """
    Retourne dict avec outcome_up, outcome_down, outcomePrices_raw
    ou None si le contrat n'est pas trouvé / pas encore résolu.
    Lève urllib.error.HTTPError pour 429.
    """
    url = f"{API_BASE}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise  # 429, 5xx → géré par l'appelant

    # data peut être un dict (event) ou une liste d'events
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]

    # Cherche les markets UP/DOWN dans l'event
    markets = data.get('markets') or []
    outcome_up   = None
    outcome_down = None
    prices_raw   = None

    for mkt in markets:
        out_names = mkt.get('outcomes') or '[]'
        if isinstance(out_names, str):
            try: out_names = json.loads(out_names)
            except: out_names = []
        prices_str = mkt.get('outcomePrices') or '[]'
        if isinstance(prices_str, str):
            try: prices = json.loads(prices_str)
            except: prices = []
        else:
            prices = prices_str

        # Cherche les outcomes UP/DOWN dans les noms
        up_idx   = None
        down_idx = None
        for i, name in enumerate(out_names):
            nm = str(name).strip().lower()
            if nm in ('up', 'higher', 'pumps', 'pump'):
                up_idx = i
            elif nm in ('down', 'lower', 'dumps', 'dump'):
                down_idx = i

        if up_idx is None and down_idx is None:
            # Fallback : outcomePrices[0]=UP, [1]=DOWN (convention Polymarket BTC/ETH)
            if len(prices) == 2:
                up_idx, down_idx = 0, 1

        if up_idx is not None and len(prices) > up_idx:
            outcome_up   = float(prices[up_idx])
        if down_idx is not None and len(prices) > down_idx:
            outcome_down = float(prices[down_idx])
        prices_raw = json.dumps(prices)
        break  # premier market suffit (slug = 1 event = 1 market)

    if outcome_up is None and outcome_down is None:
        # Essayer directement sur l'event (cas event simple sans markets)
        op = data.get('outcomePrices')
        if op:
            try:
                prices = json.loads(op) if isinstance(op, str) else op
                if len(prices) >= 2:
                    outcome_up   = float(prices[0])
                    outcome_down = float(prices[1])
                    prices_raw   = json.dumps(prices)
            except:
                pass

    if outcome_up is None:
        return None

    return {
        'outcome_up':        outcome_up,
        'outcome_down':      outcome_down,
        'outcomePrices_raw': prices_raw,
    }


# ── Boucle principale ──────────────────────────────────────────────────────────
def run(rate: float, reset: bool, dry_run: bool):
    now_ts = time.time()

    # Charger tous les contrats
    print("Chargement des contracts depuis les CSV...")
    all_contracts = []
    for asset in ASSET_CSV_SOURCES:
        ctrs = gen_contracts_for_asset(asset)
        # Filtrer : uniquement les contrats déjà fermés (ce_ts < now - 60s de marge)
        ctrs = [c for c in ctrs if c['ce_ts'] < now_ts - 60]
        sources = [p.name for p in ASSET_CSV_SOURCES[asset] if p.exists()]
        print(f"  {asset.upper()}: {len(ctrs)} contrats fermés  (sources: {', '.join(sources)})")
        all_contracts.extend(ctrs)

    all_contracts.sort(key=lambda c: c['open_ts'])
    print(f"Total : {len(all_contracts)} contrats à scraper\n")

    # Charger les slugs déjà scrapés
    if reset and OUT_CSV.exists():
        OUT_CSV.unlink()
        print(f"Reset : {OUT_CSV} supprimé.\n")

    done_slugs = load_existing(OUT_CSV)
    pending    = [c for c in all_contracts if c['slug'] not in done_slugs]
    print(f"Déjà scrapés : {len(done_slugs)}  |  Restants : {len(pending)}")

    if not pending:
        print("Rien à scraper.")
        return

    if dry_run:
        print(f"\n[DRY RUN] Premiers slugs qui seraient scrapés :")
        for c in pending[:10]:
            print(f"  {c['slug']}")
        return

    # Ouvre le CSV en append
    file_exists = OUT_CSV.exists()
    out_f = open(OUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()

    limiter = RateLimiter(rate)
    backoff  = 1.0        # secondes de backoff en cas de 429
    ok_count = 0
    miss_count = 0
    err_count  = 0
    not_found  = 0
    start_t    = time.monotonic()

    try:
        for i, c in enumerate(pending):
            limiter.acquire()

            slug = c['slug']
            attempts = 0
            result   = None

            while attempts < 5:
                try:
                    result = fetch_outcome(slug)
                    backoff = max(1.0, backoff * 0.9)  # récupération progressive
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait = backoff * (2 ** attempts)
                        print(f"\n  [429] Rate limit hit. Attente {wait:.1f}s ...")
                        time.sleep(wait)
                        backoff = min(backoff * 2, 60.0)
                        attempts += 1
                    elif e.code >= 500:
                        wait = 2.0 * (attempts + 1)
                        print(f"\n  [5xx] Erreur serveur. Attente {wait:.0f}s ...")
                        time.sleep(wait)
                        attempts += 1
                    else:
                        err_count += 1
                        break
                except Exception as ex:
                    wait = 2.0 * (attempts + 1)
                    print(f"\n  [ERR] {slug}: {ex}. Attente {wait:.0f}s ...")
                    time.sleep(wait)
                    attempts += 1

            if result is None:
                not_found += 1
                # On écrit quand même pour ne pas re-scraper
                row = {
                    'slug':        slug,
                    'asset':       c['asset'],
                    'tf':          c['tf'],
                    'open_ts':     c['open_ts'],
                    'ce_ts':       c['ce_ts'],
                    'outcome_up':  '',
                    'outcome_down': '',
                }
            else:
                ok_count += 1
                row = {
                    'slug':        slug,
                    'asset':       c['asset'],
                    'tf':          c['tf'],
                    'open_ts':     c['open_ts'],
                    'ce_ts':       c['ce_ts'],
                    'outcome_up':  result['outcome_up'],
                    'outcome_down': result['outcome_down'],
                }

            writer.writerow(row)
            out_f.flush()

            # Affichage de progression
            elapsed = time.monotonic() - start_t
            speed   = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (len(pending) - i - 1) / speed if speed > 0 else 0
            pct     = 100 * (i + 1) / len(pending)

            if result:
                up_str = f"UP={result['outcome_up']:.2f} DN={result['outcome_down']:.2f}"
            else:
                up_str = "NOT FOUND"

            print(
                f"\r[{i+1:5d}/{len(pending)}] {pct:5.1f}%  {speed:5.1f} req/s  "
                f"ETA {eta/60:.1f}min  ok={ok_count} miss={not_found}  "
                f"{slug[:45]:<45} {up_str}",
                end='', flush=True
            )

    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur.")
    finally:
        out_f.close()

    elapsed = time.monotonic() - start_t
    print(f"\n\n{'='*60}")
    print(f"Terminé en {elapsed:.1f}s")
    print(f"  OK          : {ok_count}")
    print(f"  Not found   : {not_found}")
    print(f"  Erreurs     : {err_count}")
    print(f"  CSV         : {OUT_CSV}")


# ── Fix h1 : recalcule slugs + re-fetch uniquement les h1 vides ───────────────
def fix_h1(rate: float):
    """
    Lit le CSV existant, trouve les h1 avec outcome vide,
    recalcule le bon slug (avec/sans annee selon la date),
    re-fetch et met à jour le CSV en place.
    """
    if not OUT_CSV.exists():
        print("settlement.csv introuvable.")
        return

    df = pd.read_csv(OUT_CSV)
    mask = (df['tf'] == '1h') & (df['outcome_up'].isna() | (df['outcome_up'].astype(str) == ''))
    missing = df[mask].copy()
    print(f"H1 vides : {len(missing)} lignes")
    if missing.empty:
        print("Rien à corriger.")
        return

    limiter = RateLimiter(rate)
    ok = 0; still_missing = 0
    for idx, row in missing.iterrows():
        open_ts = int(row['open_ts'])
        asset   = str(row['asset']).lower()
        new_slug = make_h1_slug(asset, open_ts)

        if new_slug == row['slug']:
            # Slug inchangé → vraiment pas sur Polymarket, on passe
            still_missing += 1
            continue

        limiter.acquire()
        result = None
        for attempt in range(3):
            try:
                result = fetch_outcome(new_slug)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** attempt * 2)
                else:
                    break
            except Exception:
                time.sleep(2)

        if result:
            df.at[idx, 'slug']        = new_slug
            df.at[idx, 'outcome_up']  = result['outcome_up']
            df.at[idx, 'outcome_down'] = result['outcome_down']
            ok += 1
            status = f"UP={result['outcome_up']:.0f} DN={result['outcome_down']:.0f}"
        else:
            df.at[idx, 'slug'] = new_slug  # au moins le slug est bon
            still_missing += 1
            status = "NOT FOUND"

        print(f"\r  [{ok+still_missing}/{len(missing)}]  {new_slug[:55]:<55}  {status}", end='', flush=True)

    print()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nOK : {ok}  Toujours vides : {still_missing}  CSV mis a jour : {OUT_CSV}")


# ── Fix pre-DST : corrige les H1 dont le slug était calculé avec UTC-4 en hiver ─
# DST 2026 : 8 mars 2026 à 02:00 EST = 07:00 UTC → open_ts < 1772953200 sont affectés
DST_2026_UTC = 1772953200  # 2026-03-08 07:00 UTC

def fix_pre_dst(rate: float):
    """
    Re-fetche uniquement les H1 avec open_ts < DST_2026_UTC dont le slug change
    après correction EST/EDT. Met à jour slug + outcome dans settlement.csv.
    """
    if not OUT_CSV.exists():
        print("settlement.csv introuvable.")
        return

    df = pd.read_csv(OUT_CSV)
    mask = (df['tf'] == '1h') & (df['open_ts'].astype(int) < DST_2026_UTC)
    candidates = df[mask].copy()
    print(f"H1 pré-DST : {len(candidates)} lignes à vérifier")

    to_fix = []
    for idx, row in candidates.iterrows():
        open_ts  = int(row['open_ts'])
        asset    = str(row['asset']).lower()
        new_slug = make_h1_slug(asset, open_ts)
        if new_slug != row['slug']:
            to_fix.append((idx, row, new_slug))

    print(f"Slugs incorrects : {len(to_fix)}")
    if not to_fix:
        print("Rien à corriger.")
        return

    limiter  = RateLimiter(rate)
    ok = 0; still_missing = 0
    for i, (idx, row, new_slug) in enumerate(to_fix):
        limiter.acquire()
        result = None
        for attempt in range(3):
            try:
                result = fetch_outcome(new_slug)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** attempt * 2)
                else:
                    break
            except Exception:
                time.sleep(2)

        df.at[idx, 'slug'] = new_slug
        if result:
            df.at[idx, 'outcome_up']   = result['outcome_up']
            df.at[idx, 'outcome_down'] = result['outcome_down']
            ok += 1
            status = f"UP={result['outcome_up']:.0f} DN={result['outcome_down']:.0f}"
        else:
            still_missing += 1
            status = "NOT FOUND"

        print(f"\r  [{i+1}/{len(to_fix)}]  {new_slug[:55]:<55}  {status}", end='', flush=True)

    print()
    df.to_csv(OUT_CSV, index=False)
    print(f"\nOK : {ok}  Toujours vides : {still_missing}  CSV mis a jour : {OUT_CSV}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Scrape Gamma API pour les vrais outcomes BTC/ETH")
    ap.add_argument('--rate',        type=float, default=3.0,
                    help='Requetes par seconde (defaut: 3.0)')
    ap.add_argument('--reset',       action='store_true',
                    help='Repart de zero (efface settlement.csv existant)')
    ap.add_argument('--dry-run',     action='store_true',
                    help='Affiche les slugs sans scraper')
    ap.add_argument('--fix-h1',      action='store_true',
                    help='Recalcule et re-fetch uniquement les h1 avec outcome vide')
    ap.add_argument('--fix-pre-dst', action='store_true',
                    help='Corrige les H1 pré-DST (14 fev → 8 mars) dont le slug était décalé de 1h')
    args = ap.parse_args()

    print(f"Rate limit : {args.rate} req/s")
    print(f"Output     : {OUT_CSV}\n")

    if args.fix_pre_dst:
        fix_pre_dst(rate=args.rate)
    elif args.fix_h1:
        fix_h1(rate=args.rate)
    else:
        run(rate=args.rate, reset=args.reset, dry_run=args.dry_run)

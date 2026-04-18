"""
Regenere les charts ETH depuis un fichier de resultats existant.
Usage:
  python regen_charts_eth.py
  python regen_charts_eth.py --results path/to/eth_curve_results_v1_*.txt --n-top 12
"""
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eth_curve_optimizer_v1 import DEFAULT_CSV_PATHS
from chart_utils import generate_charts


def parse_results(txt_path, section="TOP 30 GLOBAL", n_top=12):
    text = Path(txt_path).read_text(encoding="utf-8")
    start = text.find(f">>> {section}")
    if start == -1:
        raise ValueError(f"Section '{section}' introuvable dans {txt_path}")
    end = text.find(">>>", start + 4)
    block = text[start:end] if end != -1 else text[start:]

    configs = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("─") or line.startswith(">>>"):
            continue
        if "|" not in line:
            continue

        left, right = line.split("|", 1)
        m = re.match(r"^(\d+)\s+(\S+)\s+(.+)$", left.strip())
        if not m:
            continue

        label = m.group(2)
        rest  = m.group(3).strip()

        slope     = float(re.search(r"sl=([\d.]+)", rest).group(1))
        intercept = float(re.search(r"int=([\d.]+)", rest).group(1))
        cap       = float(re.search(r"cap=([\d.]+)", rest).group(1))
        cb_s      = re.search(r"cb=(\S+)", rest).group(1).strip()
        max_losses_cb = int(cb_s) if re.fullmatch(r"\d+", cb_s) else None

        trades_m = re.search(r"T=\s*(\d+)", right)
        maxdd_m  = re.search(r"maxDD=\$\s*([\d.]+)", right)
        rf_m     = re.search(r"RF=\s*([\d.]+)x", right)

        trades = int(trades_m.group(1))   if trades_m else 0
        max_dd = float(maxdd_m.group(1))  if maxdd_m  else 0.0
        rf     = float(rf_m.group(1))     if rf_m     else 0.0

        cfg = dict(
            label=label, curve="linear",
            slope=slope, intercept=intercept, intercept_mode="floor",
            A_exp=None, tau=None,
            eq_cap=cap, max_losses_cb=max_losses_cb,
            vol_lb_h=None, vol_thresh=None,
            trades=trades, max_dd=max_dd, rf=rf,
        )

        if "cnet" in label:
            m2 = re.search(r"cnet(\d+)c>\$([\d.]+)", rest)
            if m2:
                cfg["cnet_n"]      = int(m2.group(1))
                cfg["cnet_thresh"] = float(m2.group(2))
        elif "vrs" in label:
            m2 = re.search(r"vrs([\d.]+)h b=([\d.]+) g=([\d.]+)-([\d.]+)", rest)
            if m2:
                cfg["vrs_enabled"] = True
                cfg["vrs_lb"]      = float(m2.group(1))
                cfg["vrs_base"]    = float(m2.group(2))
                cfg["vrs_g_min"]   = float(m2.group(3))
                cfg["vrs_g_max"]   = float(m2.group(4))
        elif "net" in label:
            cfg["vol_type"] = "net"
            m2 = re.search(r"net([\d.]+)h>\$([\d.]+)", rest)
            if m2:
                cfg["vol_lb_h"]   = float(m2.group(1))
                cfg["vol_thresh"] = float(m2.group(2))
        elif "trend" in label:
            cfg["vol_type"] = "trend"
            m2 = re.search(r"tre([\d.]+)h>([\d.]+)", rest)
            if m2:
                cfg["vol_lb_h"]   = float(m2.group(1))
                cfg["vol_thresh"] = float(m2.group(2))

        configs.append(cfg)
        if len(configs) >= n_top:
            break

    return configs


def find_latest_results(god_curve_dir):
    candidates = sorted(
        Path(god_curve_dir).glob("god_curve_eth_v1_*/eth_curve_results_v1_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Aucun fichier de resultats ETH trouve.")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Regenere les charts ETH depuis un fichier de resultats.")
    parser.add_argument("--results", default=None, help="Chemin vers eth_curve_results_v1_*.txt (auto-detect si omis)")
    parser.add_argument("--out",     default=None, help="Dossier de sortie charts (defaut: sous-dossier charts/ du run)")
    parser.add_argument("--csv",     nargs="+",    default=None)
    parser.add_argument("--n-top",   type=int,     default=12)
    parser.add_argument("--section", default="TOP 30 GLOBAL")
    args = parser.parse_args()

    god_curve_dir = Path(__file__).resolve().parent

    results_path = Path(args.results) if args.results else find_latest_results(god_curve_dir)
    out_dir      = Path(args.out) if args.out else results_path.parent / "charts"
    csv_paths    = args.csv if args.csv else DEFAULT_CSV_PATHS

    print(f"Resultats : {results_path}")
    print(f"Charts    : {out_dir}")
    print(f"CSV       : {csv_paths}")
    print(f"Section   : {args.section}  (top {args.n_top})\n")

    configs = parse_results(results_path, section=args.section, n_top=args.n_top)
    print(f"{len(configs)} configs parsees :")
    for i, c in enumerate(configs):
        print(f"  [{i+1:2d}] {c['label']:>12s}  sl={c['slope']:.3f} int={c['intercept']:.1f}  cap={c['eq_cap']:.2f}  RF={c['rf']:.1f}x")

    generate_charts(
        configs,
        csv_paths,
        str(out_dir),
        n_top=len(configs),
        title_prefix="ETH God Curve v1",
        vol_slope_ref_key="vol_1h_range",
        asset="eth",
    )


if __name__ == "__main__":
    main()

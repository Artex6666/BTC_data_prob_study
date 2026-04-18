from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_utils import simulate_market_trade_with_exit
from extract_ordresmarkets_features import extract_asset
from ordresmarkets_common import OUTPUT_DIR, build_report_contract_map, list_market_reports


def _quantiles(values: pd.Series, probs: list[float]) -> list[float]:
    if values.empty:
        return [0.0 for _ in probs]
    return [float(values.quantile(p)) for p in probs]


def _load_features(asset: str, sample_every_s: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"features_{asset}.csv"
    if not path.exists():
        extract_asset(asset, sample_every_s=sample_every_s)
    return pd.read_csv(path)


def _rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if rule.get("require_aligned"):
        mask &= df["aligned"] == 1
    if rule.get("require_recross"):
        mask &= df["recross"] == 1
    mask &= df["signed_open_pct"].abs() >= float(rule["abs_open_pct_min"])
    mask &= df["signed_move_30s_pct"].abs() >= float(rule["abs_move_30s_pct_min"])
    mask &= df["ask_c"] <= float(rule["ask_c_max"])
    mask &= df["elapsed_frac"] >= float(rule["elapsed_frac_min"])
    mask &= df["elapsed_frac"] <= float(rule["elapsed_frac_max"])
    return mask


def derive_trigger_rules(df: pd.DataFrame) -> dict[str, dict]:
    rules: dict[str, dict] = {}
    for (asset, tf_slug, side), sub in df.groupby(["asset", "tf_slug", "side"]):
        buys = sub[sub["buy_label"] == 1].copy()
        if buys.empty:
            continue

        pos_weights = np.maximum(buys["buy_cost_usd"].to_numpy(dtype=float), 1.0)
        aligned_share = float(np.average(buys["aligned"], weights=pos_weights))
        recross_share = float(np.average(buys["recross"], weights=pos_weights))
        abs_open_q = buys["signed_open_pct"].abs()
        abs_move_q = buys["signed_move_30s_pct"].abs()
        ask_q = buys["ask_c"]
        elapsed_q = buys["elapsed_frac"]

        aligned_options = [aligned_share >= 0.65]
        recross_options = [recross_share >= 0.60]
        abs_open_grid = sorted(set(_quantiles(abs_open_q, [0.5, 0.7])))
        abs_move_grid = sorted(set(_quantiles(abs_move_q, [0.5, 0.7])))
        ask_grid = sorted(set(_quantiles(ask_q, [0.6, 0.75])))
        elapsed_min_grid = sorted(set(_quantiles(elapsed_q, [0.2, 0.3])))
        elapsed_max_grid = sorted(set(_quantiles(elapsed_q, [0.85, 0.95])))

        best_score = -1.0
        best_rule = None

        y = sub["buy_label"].to_numpy(dtype=int)
        for require_aligned in aligned_options:
            for require_recross in recross_options:
                for abs_open_min in abs_open_grid:
                    for abs_move_min in abs_move_grid:
                        for ask_c_max in ask_grid:
                            for elapsed_frac_min in elapsed_min_grid:
                                for elapsed_frac_max in elapsed_max_grid:
                                    if elapsed_frac_max <= elapsed_frac_min:
                                        continue
                                    rule = {
                                        "require_aligned": bool(require_aligned),
                                        "require_recross": bool(require_recross),
                                        "abs_open_pct_min": max(0.0, float(abs_open_min)),
                                        "abs_move_30s_pct_min": max(0.0, float(abs_move_min)),
                                        "ask_c_max": float(ask_c_max),
                                        "elapsed_frac_min": max(0.0, float(elapsed_frac_min)),
                                        "elapsed_frac_max": min(1.0, float(elapsed_frac_max)),
                                    }
                                    pred = _rule_mask(sub, rule).to_numpy()
                                    tp = int(((pred) & (y == 1)).sum())
                                    fp = int(((pred) & (y == 0)).sum())
                                    fn = int(((~pred) & (y == 1)).sum())
                                    if tp == 0 or tp + fp == 0:
                                        continue
                                    precision = tp / (tp + fp)
                                    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                                    pred_ratio = pred.mean()
                                    score = precision * np.sqrt(max(recall, 1e-9))
                                    if pred_ratio > 0.20:
                                        score *= 0.5
                                    if pred_ratio < 0.01:
                                        score *= 0.7
                                    if score > best_score:
                                        best_score = score
                                        best_rule = {
                                            **rule,
                                            "score": float(score),
                                            "precision": float(precision),
                                            "recall": float(recall),
                                            "predicted_rows": int(pred.sum()),
                                            "buy_rows": int(len(buys)),
                                            "entry_size_usd": float(buys["buy_cost_usd"].median()),
                                        }

        if best_rule is not None:
            rules[f"{asset}|{tf_slug}|{side}"] = best_rule
    return rules


def derive_exit_policies(df: pd.DataFrame, contract_map: dict[str, dict]) -> dict[str, dict]:
    policies: dict[str, dict] = {}
    hold_grid = {
        "m5": [None, 60, 120],
        "m15": [None, 180, 300],
        "h1": [None, 900, 1800],
    }
    tp_grid = [None, 8, 12]
    sl_grid = [None, 6, 10]

    pos = df[df["buy_label"] == 1].copy()
    if pos.empty:
        return policies

    for (asset, tf_slug), sub in pos.groupby(["asset", "tf_slug"]):
        sub = sub.sort_values("buy_cost_usd", ascending=False).head(120)
        best_key = None
        best_score = -1e18
        for tp_c in tp_grid:
            for sl_c in sl_grid:
                for max_hold_s in hold_grid[tf_slug]:
                    pnls = []
                    for row in sub.itertuples(index=False):
                        contract = contract_map.get(row.market_key)
                        if contract is None:
                            continue
                        trade = simulate_market_trade_with_exit(
                            contract,
                            side=row.side.upper(),
                            entry_j=int(row.contract_j),
                            size_usd=100.0,
                            take_profit_c=tp_c,
                            stop_loss_c=sl_c,
                            max_hold_s=max_hold_s,
                        )
                        if trade is not None:
                            pnls.append(trade["pnl"])
                    if not pnls:
                        continue
                    score = float(np.mean(pnls))
                    if score > best_score:
                        best_score = score
                        best_key = {
                            "take_profit_c": tp_c,
                            "stop_loss_c": sl_c,
                            "max_hold_s": max_hold_s,
                            "mean_pnl_per_100usd": score,
                        }
        if best_key is not None:
            policies[f"{asset}|{tf_slug}"] = best_key
    return policies


def backtest_asset(asset: str, df: pd.DataFrame, rules: dict[str, dict], policies: dict[str, dict]) -> dict:
    contract_map = build_report_contract_map(asset)
    meta_map = {meta.market_key: meta for meta in list_market_reports(asset=asset)}

    candidate_trades: list[dict] = []
    market_rows: list[dict] = []

    for market_key, sub in df.groupby("market_key"):
        meta = meta_map.get(market_key)
        contract = contract_map.get(market_key)
        if meta is None or contract is None:
            continue

        market_candidate_pnl = 0.0
        market_candidate_spent = 0.0
        market_trade_count = 0

        for side in ("Up", "Down"):
            side_rows = sub[sub["side"] == side].sort_values("contract_j")
            rule_key = f"{asset}|{meta.tf_slug}|{side}"
            rule = rules.get(rule_key)
            if rule is None:
                continue
            triggers = side_rows[_rule_mask(side_rows, rule)]
            if triggers.empty:
                continue
            first = triggers.iloc[0]
            policy = policies.get(rule_key) or policies.get(f"{asset}|{meta.tf_slug}", {})
            size_usd = max(1.0, float(rule.get("entry_size_usd", 100.0)))
            trade = simulate_market_trade_with_exit(
                contract,
                side=side.upper(),
                entry_j=int(first["contract_j"]),
                size_usd=size_usd,
                take_profit_c=policy.get("take_profit_c"),
                stop_loss_c=policy.get("stop_loss_c"),
                max_hold_s=policy.get("max_hold_s"),
            )
            if trade is None:
                continue
            trade_row = {
                "asset": asset,
                "market_key": market_key,
                "market_title": meta.market_title,
                "tf_slug": meta.tf_slug,
                "side": side,
                "signal_ts": first["timestamp"],
                "signed_open_pct": float(first["signed_open_pct"]),
                "signed_move_30s_pct": float(first["signed_move_30s_pct"]),
                "ask_c": float(first["ask_c"]),
                "rule_key": rule_key,
                **trade,
            }
            candidate_trades.append(trade_row)
            market_candidate_pnl += float(trade["pnl"])
            market_candidate_spent += float(trade["size_usd"])
            market_trade_count += 1

        market_rows.append(
            {
                "asset": asset,
                "market_key": market_key,
                "market_title": meta.market_title,
                "tf_slug": meta.tf_slug,
                "ce_ts": meta.ce_ts.isoformat(),
                "bot_pnl": meta.report_pnl,
                "bot_spent": meta.report_spent,
                "candidate_pnl": market_candidate_pnl,
                "candidate_spent": market_candidate_spent,
                "candidate_trade_count": market_trade_count,
                "delta_pnl": market_candidate_pnl - meta.report_pnl,
            }
        )

    trades_df = pd.DataFrame(candidate_trades)
    market_df = pd.DataFrame(market_rows).sort_values("ce_ts")

    return {
        "trades": trades_df,
        "markets": market_df,
    }


def _equity_series(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.array([], dtype=float)
    return values.to_numpy(dtype=float).cumsum()


def _write_asset_outputs(asset: str, result: dict, rules: dict, policies: dict) -> dict:
    trades_df = result["trades"]
    markets_df = result["markets"]

    trades_csv = OUTPUT_DIR / f"candidate_trades_{asset}.csv"
    markets_csv = OUTPUT_DIR / f"market_comparison_{asset}.csv"
    trades_df.to_csv(trades_csv, index=False)
    markets_df.to_csv(markets_csv, index=False)

    bot_pnl = float(markets_df["bot_pnl"].sum()) if not markets_df.empty else 0.0
    bot_spent = float(markets_df["bot_spent"].sum()) if not markets_df.empty else 0.0
    cand_pnl = float(markets_df["candidate_pnl"].sum()) if not markets_df.empty else 0.0
    cand_spent = float(markets_df["candidate_spent"].sum()) if not markets_df.empty else 0.0
    trade_wr = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else 0.0

    summary = {
        "asset": asset,
        "covered_markets": int(len(markets_df)),
        "candidate_trades": int(len(trades_df)),
        "bot_total_pnl": bot_pnl,
        "bot_total_spent": bot_spent,
        "bot_roi_pct": (bot_pnl / bot_spent * 100.0) if bot_spent > 0 else 0.0,
        "candidate_total_pnl": cand_pnl,
        "candidate_total_spent": cand_spent,
        "candidate_roi_pct": (cand_pnl / cand_spent * 100.0) if cand_spent > 0 else 0.0,
        "candidate_trade_wr": trade_wr,
        "rules": {k: v for k, v in rules.items() if k.startswith(f"{asset}|")},
        "exit_policies": {k: v for k, v in policies.items() if k.startswith(f"{asset}|")},
    }
    summary_path = OUTPUT_DIR / f"comparison_summary_{asset}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    txt_lines = [
        f"Asset: {asset}",
        f"Markets covered: {summary['covered_markets']}",
        f"Bot pnl/spent/roi: {bot_pnl:.2f} / {bot_spent:.2f} / {summary['bot_roi_pct']:.2f}%",
        f"Candidate pnl/spent/roi: {cand_pnl:.2f} / {cand_spent:.2f} / {summary['candidate_roi_pct']:.2f}%",
        f"Candidate trades: {summary['candidate_trades']}",
        f"Candidate trade WR: {trade_wr * 100.0:.2f}%",
        "",
        "Derived rules:",
    ]
    for key, value in summary["rules"].items():
        txt_lines.append(f"{key}: {value}")
    txt_lines.append("")
    txt_lines.append("Derived exit policies:")
    for key, value in summary["exit_policies"].items():
        txt_lines.append(f"{key}: {value}")
    (OUTPUT_DIR / f"comparison_summary_{asset}.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    plt.figure(figsize=(10, 5))
    if not markets_df.empty:
        x = np.arange(len(markets_df))
        plt.plot(x, _equity_series(markets_df["bot_pnl"]), label="Bot reel", linewidth=2.0)
        plt.plot(x, _equity_series(markets_df["candidate_pnl"]), label="Strategie derivee", linewidth=2.0)
    plt.title(f"OrdresMarkets compare - {asset.upper()}")
    plt.xlabel("Marches couverts")
    plt.ylabel("PnL cumule ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"equity_compare_{asset}.png", dpi=150)
    plt.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a strategy derived from ordresMarkets reports")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["btc", "eth"],
        choices=["btc", "eth"],
        help="Assets to process.",
    )
    parser.add_argument(
        "--sample-every-s",
        type=int,
        default=5,
        help="Sampling interval to use if feature CSVs must be rebuilt.",
    )
    args = parser.parse_args()

    all_rules: dict[str, dict] = {}
    all_policies: dict[str, dict] = {}
    summaries: list[dict] = []

    for asset in args.assets:
        df = _load_features(asset, sample_every_s=max(1, int(args.sample_every_s)))
        if df.empty:
            print(f"{asset}: no data")
            continue
        print(f"{asset}: deriving trigger rules on {len(df)} rows")
        rules = derive_trigger_rules(df)
        print(f"{asset}: building covered-contract map")
        contract_map = build_report_contract_map(asset)
        print(f"{asset}: deriving exit policies on {len(contract_map)} contracts")
        policies = derive_exit_policies(df, contract_map)
        all_rules.update(rules)
        all_policies.update(policies)

        print(f"{asset}: running comparison backtest")
        result = backtest_asset(asset, df, rules, policies)
        summary = _write_asset_outputs(asset, result, rules, policies)
        summaries.append(summary)
        print(
            f"{asset}: candidate pnl {summary['candidate_total_pnl']:.2f} "
            f"vs bot {summary['bot_total_pnl']:.2f}"
        )

    (OUTPUT_DIR / "derived_rules.json").write_text(json.dumps(all_rules, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "derived_exit_policies.json").write_text(json.dumps(all_policies, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "all_assets_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

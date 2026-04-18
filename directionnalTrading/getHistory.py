import sys
import time
import json
import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.ticker as ticker
import os
import re
import shutil

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
STYLES = {
    ("Buy", "Up"):   ("#008f00", "x", "Buy YES"),   # strong green
    ("Sell", "Up"):  ("#00c800", "o", "Sell YES"),  # vivid lime
    ("Buy", "Down"): ("#d000d0", "x", "Buy NO"),    # strong magenta
    ("Sell", "Down"):("#d40000", "o", "Sell NO")    # strong red
}

USERS = {
    "perso": "0x502b532f55778c0985846dc53a9bc71eb4f33b15", # -> MOI
    "petitJoueur":"0x6b1bdf3c115b85083dd41feae4daf7a634531923", # -> ALTERNANCE UP /DOWN RENTABLE MAIS EN BT ON COMPREND PAS COMMENT IL FAIT
    "QtechAllCryptos":"0x537494c54dee9162534675712f2e625c9713042e", # -> en perte actuellement 
    "luckyNumberThirteen":"0xd4253cf046d1e7080c2bab443737710d55a9c0e4", # -> il envoie des ordres markets
    "Axia":"0x18cda925145a03cd1650ad1036f6b152ece1d8d2",
    "botToujoursDownWTF": "0xda8f19a077db26a7c640bd5055e0a750665f83d9", # toujours down, doit profiter de la tendance bear
    "ordresMarkets2": "0xf3a710078c35b891cc0ce98c1badaa023a4f2fd8", #ordres markets
    "ordresMarkets": "0x04283f2fef49d70d8c55ab240450d17a65bf85b1"  # il rentre market avec des montatns de salopard wtf

}

#OLD  USERS 
#     "gabagool": "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",
#     "boug": "0x5fb40CA38E0dd2cC496fF1C2A1CcFaCce9b16e19",
#     "vendeur":"0xf247584e41117bbbe4cc06e4d2c95741792a5216", # -> COMPRENDS PAS OCMMENT IL FONCTIONNE, ACHAT REVENTE PARFOIS OUI, PARFOIS NON
#     "BotTwitter2":"0xd4583c4704a8c2e416f0e7fa5b763f92f0291733", # -> TRES PEU DE VOLUME, IN EN FIN DE CONTRAT A 99 AVEC ENORME CAPITAL
#     "BotTwitter3":"0xd0d6053c3c37e727402d84c14069780d360993aa", # -> important mais ne trade plus BTC (va savoir pourquoi)?
#     "directionnelSometimes":"0x0f863d92dd2b960e3eb6a23a35fd92a91981404e", -> a quitté les cryptos
#     "Anon":"0xad6f2d2150a3991ac6cd16df8fb5e2807608b2fb",
#     "bees":"0x61276aba49117fd9299707d5d573652949d5c977", 
#     "distinctBaguette":"0xe00740bce98a594e26861838885ab310ec3b548c",









SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
TRADES_URL = "https://data-api.polymarket.com/trades"
PRICE_RESOLUTION_THRESHOLD = 0.5

# ---------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------
def write_stats_report(
    report_path,
    target_market,
    resolved_side,
    trade_count,
    remaining_yes,
    remaining_no,
    final_value,
    total_spent,
    pnl,
    yes_buy_sh,
    yes_buy_cost,
    yes_sell_sh,
    yes_sell_cost,
    no_buy_sh,
    no_buy_cost,
    no_sell_sh,
    no_sell_cost,
    cum_yes_total,
    cum_no_total,
    cum_yes_cost_total,
    cum_no_cost_total,
    yes_curve,
    no_curve,
    net_curve,
    yes_sh_curve,
    no_sh_curve,
    net_sh_curve,
    prices,
    trades,
):
    # Safety checks for empty data
    if len(yes_curve) > 0:
        yes_peak_idx = int(np.argmax(yes_curve))
        no_peak_idx = int(np.argmax(no_curve))
        yes_sh_peak_idx = int(np.argmax(yes_sh_curve))
        no_sh_peak_idx = int(np.argmax(no_sh_curve))
        
        yes_peak_val = yes_curve[yes_peak_idx]
        no_peak_val = no_curve[no_peak_idx]
        yes_sh_peak_val = yes_sh_curve[yes_sh_peak_idx]
        no_sh_peak_val = no_sh_curve[no_sh_peak_idx]
        
        final_yes_exp = yes_curve[-1]
        final_yes_sh = yes_sh_curve[-1]
        final_no_exp = no_curve[-1]
        final_no_sh = no_sh_curve[-1]
        final_net_exp = net_curve[-1]
        final_net_sh = net_sh_curve[-1]
    else:
        yes_peak_idx = no_peak_idx = 0
        yes_sh_peak_idx = no_sh_peak_idx = 0
        yes_peak_val = no_peak_val = 0
        yes_sh_peak_val = no_sh_peak_val = 0
        final_yes_exp = final_yes_sh = 0
        final_no_exp = final_no_sh = 0
        final_net_exp = final_net_sh = 0

    def _format_ts_ms(ts):
        t = float(ts)
        dt = datetime.datetime.fromtimestamp(t, tz=datetime.UTC)
        ms = int(round((t % 1) * 1000))
        if ms != 0:
            return dt.strftime('%Y-%m-%d %H:%M:%S') + f'.{ms:03d}Z'
        return dt.strftime('%Y-%m-%d %H:%M:%S') + 'Z'

    # Calculate time range
    start_time = "N/A"
    end_time = "N/A"
    if trades:
        start_time = _format_ts_ms(trades[0]['timestamp'])
        end_time = _format_ts_ms(trades[-1]['timestamp'])
    
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0

    avg_yes_buy = yes_buy_cost / yes_buy_sh if yes_buy_sh > 0 else 0
    avg_no_buy = no_buy_cost / no_buy_sh if no_buy_sh > 0 else 0
    combined_avg = avg_yes_buy + avg_no_buy

    lines = [
        f"MARKET: {target_market}",
        f"RESOLUTION: {resolved_side}",
        f"TRADES: {trade_count}",
        f"TIME RANGE: {start_time} to {end_time}",
        f"PRICE RANGE: {min_price:.2f} - {max_price:.2f}",
        "",
        "--- Position at resolution ---",
        f"Remaining YES shares: {remaining_yes:.2f}",
        f"Remaining NO shares:  {remaining_no:.2f}",
        f"Final value: $ {final_value:.2f}",
        f"Total spent (net exposure): $ {total_spent:.2f}",
        f"FINAL PNL: $ {pnl:.2f}",
        "",
        "--- Buy/Sell totals ---",
        f"YES buys:  {yes_buy_sh:.2f} sh / $ {yes_buy_cost:.2f} (Avg: $ {avg_yes_buy:.4f})",
        f"YES sells: {yes_sell_sh:.2f} sh / $ {yes_sell_cost:.2f}",
        f"NO buys:   {no_buy_sh:.2f} sh / $ {no_buy_cost:.2f} (Avg: $ {avg_no_buy:.4f})",
        f"NO sells:  {no_sell_sh:.2f} sh / $ {no_sell_cost:.2f}",
        f"Sum of Avgs (YES+NO): $ {combined_avg:.4f}",
        "",
        "--- Cumulative buys ---",
        f"YES cumulative: {cum_yes_total:.2f} sh / $ {cum_yes_cost_total:.2f}",
        f"NO cumulative:  {cum_no_total:.2f} sh / $ {cum_no_cost_total:.2f}",
        "",
        "--- Exposure peaks (trade index: earliest → latest) ---",
        f"YES dollar peak: $ {yes_peak_val:.2f} at trade #{yes_peak_idx + 1}",
        f"NO dollar peak:  $ {no_peak_val:.2f} at trade #{no_peak_idx + 1}",
        f"YES share peak:  {yes_sh_peak_val:.2f} sh at trade #{yes_sh_peak_idx + 1}",
        f"NO share peak:   {no_sh_peak_val:.2f} sh at trade #{no_sh_peak_idx + 1}",
        "",
        "--- Final exposure ---",
        f"YES exposure: $ {final_yes_exp:.2f} | {final_yes_sh:.2f} sh",
        f"NO exposure:  $ {final_no_exp:.2f} | {final_no_sh:.2f} sh",
        f"NET exposure: $ {final_net_exp:.2f} | {final_net_sh:.2f} sh",
    ]

    lines.append("")
    lines.append("--- Trades (Sorted by Timestamp) ---")
    lines.append("Idx | Time                | Type | Side | Price(c) |   Shares   |    Cost($)")
    lines.append("----+---------------------+------+-----+----------+------------+------------")

    for i, t in enumerate(trades):
        dt_str = _format_ts_ms(t['timestamp'])
        lines.append(
            f"{i+1:3d} | {dt_str} | {t['type']:<4} | {t['side']:<4} | "
            f"{t['price']:8.2f} | {t['shares']:10.2f} | $ {t['cost']:9.2f}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------
def search_markets(query):
    """Return list of (event, market) for all matching search results, fetching multiple pages."""
    results = []
    limit = 20 # API seems to return small batches, let's request 20
    offset = 0
    max_pages = 5 # Fetch up to 100 items total to find recent closed markets
    
    for _ in range(max_pages):
        try:
            # print(f"DEBUG: Searching '{query}' offset={offset}")
            resp = requests.get(SEARCH_URL, params={"q": query, "limit": limit, "offset": offset}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"Error searching market: {exc}")
            break

        events = data.get("events", []) if isinstance(data, dict) else []
        if not events:
            break
            
        for event in events:
            markets = event.get("markets") or []
            for market in markets:
                results.append((event, market))
        
        offset += len(events)
        
        # If we got fewer than expected, we probably hit the end
        if len(events) < 5: # The API seems to return 5 items per page in some contexts
            # Check pagination key if available, otherwise strict check?
            # Debug output showed 'pagination': {'hasMore': True, 'totalResults': 38689}
            # If hasMore is False, we stop.
            pagination = data.get("pagination", {})
            if not pagination.get("hasMore", True):
                break
                
    return results


def fetch_trades(condition_id, user_address, page_limit=500):
    """Fetch all trades for a condition/user with simple pagination."""
    all_trades = []
    offset = 0

    while True:
        params = {
            "limit": page_limit,
            "offset": offset,
            "takerOnly": "false",
            "market": condition_id,
            "user": user_address,
        }
        try:
            resp = requests.get(TRADES_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"Error fetching trades: {exc}")
            return []

        if isinstance(data, dict):
            batch = data.get("trades", [])
        elif isinstance(data, list):
            batch = data
        else:
            batch = []

        all_trades.extend(batch)
        if len(batch) < page_limit:
            break
        offset += page_limit

    return all_trades


def prompt_resolved_side(current=None):
    """Return a valid resolved side, prompting if needed."""
    if current in {"YES", "NO"}:
        return current
    while True:
        side = input("Enter resolved side (YES/NO, blank = skip): ").strip().upper()
        if not side:
            return None
        if side in {"YES", "NO"}:
            return side
        print("Please enter YES or NO.")


def normalize_resolved_arg(value):
    if not value:
        return None
    value = value.strip().upper()
    if value in {"YES", "NO", "AUTO"}:
        return value
    return None


def infer_resolved_side_from_trades(trades, threshold=PRICE_RESOLUTION_THRESHOLD):
    """Infer resolved side from the most recent trade price."""
    if not trades:
        return None, None
    latest = max(trades, key=lambda t: t.get("timestamp", 0))
    price = float(latest.get("price", 0))
    outcome = latest.get("outcome", "").lower()

    if outcome not in {"up", "down"}:
        return None, latest

    # If price >= threshold, assume resolved toward that outcome; otherwise opposite.
    if price >= threshold:
        inferred = "YES" if outcome == "up" else "NO"
    else:
        inferred = "NO" if outcome == "up" else "YES"
    return inferred, latest

def sanitize_filename(name):
    """Sanitize the string to be safe for filenames."""
    # Keep only alphanumeric, spaces, hyphens and dots
    s = re.sub(r'[^a-zA-Z0-9 \-\.]', '', name)
    # Replace spaces with underscores
    s = s.replace(' ', '_')
    return s


def parse_market_name_to_slug(market_name):
    """
    Convert market name to slug format.
    Examples:
    - "Bitcoin Up or Down – January 25, 4:00AM–4:15AM ET" -> "btc-updown-15m-1769331600"
    - "Bitcoin Up or Down – January 25, 4:00AM ET" -> "bitcoin-up-or-down-january-25-4am-et"
    """
    name_lower = market_name.lower()
    
    # Detect crypto
    crypto = None
    crypto_short = None
    if 'bitcoin' in name_lower or 'btc' in name_lower:
        crypto = 'bitcoin'
        crypto_short = 'btc'
    elif 'ethereum' in name_lower or 'eth' in name_lower:
        crypto = 'ethereum'
        crypto_short = 'eth'
    elif 'solana' in name_lower or 'sol' in name_lower:
        crypto = 'solana'
        crypto_short = 'sol'
    elif 'xrp' in name_lower or 'ripple' in name_lower:
        crypto = 'xrp'
        crypto_short = 'xrp'
    else:
        return None
    
    # Detect timeframe: M15 (has time range like "4:00AM–4:15AM" or "4:00AM-4:15AM") or H1 (single hour)
    # Check for time range pattern (two times separated by dash/en-dash)
    time_range_pattern = r'(\d+):(\d+)(AM|PM)\s*[–-]\s*(\d+):(\d+)(AM|PM)'
    has_time_range = re.search(time_range_pattern, market_name, re.IGNORECASE) is not None
    is_m15 = has_time_range
    
    # Parse date and time
    # Format examples:
    # - "January 25, 4:00AM–4:15AM ET" (M15)
    # - "January 25, 4:00AM ET" (H1)
    # - "January 25, 4:00AM-4:15AM ET" (M15 with hyphen)
    date_match = re.search(r'(\w+)\s+(\d+),?\s+(\d+):(\d+)(AM|PM)', market_name, re.IGNORECASE)
    if not date_match:
        return None
    
    month_name = date_match.group(1).lower()
    day = int(date_match.group(2))
    hour = int(date_match.group(3))
    minute = int(date_match.group(4))
    ampm = date_match.group(5).upper()
    
    # Convert to 24h
    if ampm == 'PM' and hour != 12:
        hour += 12
    elif ampm == 'AM' and hour == 12:
        hour = 0
    
    # Month name to number
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    month_num = months.get(month_name, 1)
    
    # Get current year (assume current year, or could parse from market name)
    current_year = datetime.datetime.now().year
    
    if is_m15:
        # M15 format: btc-updown-15m-{timestamp}
        # Timestamp is the start of the 15-minute window (rounded to 15min)
        # ET timezone offset: -5 hours (EST) or -4 hours (EDT)
        # For simplicity, assume ET = UTC-5 (EST, not EDT)
        # Create ET datetime (naive, assume EST)
        et_date = datetime.datetime(current_year, month_num, day, hour, minute, 0)
        # Convert ET to UTC (ET is UTC-5 for EST)
        # Note: This is approximate - DST handling would require more complexity
        utc_date = et_date + datetime.timedelta(hours=5)
        # Round to 15-minute window
        rounded_minute = (utc_date.minute // 15) * 15
        utc_date = utc_date.replace(minute=rounded_minute, second=0, microsecond=0)
        timestamp = int(utc_date.timestamp())
        return f"{crypto_short}-updown-15m-{timestamp}"
    else:
        # H1 format: bitcoin-up-or-down-january-25-4am-et
        hour_12 = hour if hour <= 12 else hour - 12
        if hour_12 == 0:
            hour_12 = 12
        ampm_str = 'am' if hour < 12 else 'pm'
        return f"{crypto}-up-or-down-{month_name}-{day}-{hour_12}{ampm_str}-et"


def parse_market_end_time(market_name):
    """
    Parses the market name to extract the END time.
    Returns a datetime object representing the end time in UTC (assuming ET input).
    """
    # Regex for M5/M15: "February 19, 5:00PM-5:05PM ET"
    # Matches: Month Day, StartTime-EndTime ET
    range_pattern = r'(\w+)\s+(\d+),?\s+\d+:\d+(?:AM|PM)\s*[–-]\s*(\d+):(\d+)(AM|PM)\s*ET'
    match = re.search(range_pattern, market_name, re.IGNORECASE)
    
    is_h1 = False
    
    if not match:
        # Try H1 Pattern (Single Time): "February 19, 4PM ET" -> Ends at 5PM
        h1_pattern = r'(\w+)\s+(\d+),?\s+(\d+)(?::(\d+))?\s*(AM|PM)\s*ET'
        match = re.search(h1_pattern, market_name, re.IGNORECASE)
        if match:
            is_h1 = True
    
    if not match:
        return None

    month_name = match.group(1)
    day = int(match.group(2))
    
    if is_h1:
        # H1: Group 3 is Hour, Group 4 is Minute (optional), Group 5 is AM/PM
        hour = int(match.group(3))
        minute = int(match.group(4)) if match.group(4) else 0
        ampm = match.group(5).upper()
        
        # Convert to 24h
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
            
        # Add 1 hour for H1 end time
        # Handle year turnover? Unlikely for this script usage but safe enough to ignore for now.
        dt = datetime.datetime(datetime.datetime.now().year, 1, 1, hour, minute)
        dt = dt + datetime.timedelta(hours=1)
        hour = dt.hour
        minute = dt.minute
        
    else:
        # Range: Group 3 is Hour, Group 4 is Minute, Group 5 is AM/PM
        hour = int(match.group(3))
        minute = int(match.group(4))
        ampm = match.group(5).upper()
        
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0

    # Parse Month
    try:
        dt_month = datetime.datetime.strptime(month_name, "%B")
        month_num = dt_month.month
    except ValueError:
        return None

    current_year = datetime.datetime.now().year
    
    try:
        # Create naive datetime from parsed components (Assuming Market Time is ET)
        market_end_naive_et = datetime.datetime(current_year, month_num, day, hour, minute)
        
        # Convert ET to UTC.
        # Assuming Standard Time (EST = UTC-5) for simplicity as user requested "2 minutes after".
        # If we are in DST, it would be UTC-4.
        # A robust solution would use pytz, but user env might not have it.
        # Let's stick to the +5 hours offset to get UTC from EST.
        # If the user is actually in EDT, this might be off by 1 hour (fetch 1 hour late or early).
        # Given "2 minutes after", fetching 1 hour + 2 mins after is safe (just slow).
        # Fetching 1 hour early (before end) is bad.
        # EST (UTC-5) is "later" in UTC than EDT (UTC-4).
        # e.g. 17:00 EST = 22:00 UTC. 17:00 EDT = 21:00 UTC.
        # If we assume EST (add 5h) but it was EDT, we calculate 22:00 UTC. But it ended at 21:00 UTC.
        # We wait until 22:02. That is 1 hour 2 mins after end. Safe.
        
        market_end_utc = market_end_naive_et + datetime.timedelta(hours=5)
        return market_end_utc
        
    except ValueError as e:
        print(f"Error constructing date: {e}")
        return None


# ---------------------------------------------------
# PLOTTING
# ---------------------------------------------------
def generate_chart(parsed, chart_path, market_title, resolved_side, 
                   yes_curve, no_curve, net_curve, 
                   yes_sh_curve, no_sh_curve, net_sh_curve,
                   yes_buy_sh, yes_buy_cost, yes_sell_sh, yes_sell_cost,
                   no_buy_sh, no_buy_cost, no_sell_sh, no_sell_cost,
                   pnl, remaining_yes, remaining_no, final_value, total_spent,
                   cum_yes, cum_no, cum_yes_cost, cum_no_cost,
                   cum_yes_total, cum_no_total, cum_yes_cost_total, cum_no_cost_total):
    
    unique_timestamps = sorted(list(set(t['timestamp'] for t in parsed)))
    ts_map = {ts: i for i, ts in enumerate(unique_timestamps)}
    x_indices = [ts_map[e['timestamp']] for e in parsed]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(16, 14.5),
        gridspec_kw={'height_ratios': [3, 1.3, 1.1, 1.1]}
    )
    fig.subplots_adjust(hspace=0.45, bottom=0.2)


    # ---------------------------------------------------
    # TOP: BUY/SELL SCATTER (GROUPED BUBBLE VIEW)
    # ---------------------------------------------------
    grouped_trades = {}
    for i, e in enumerate(parsed):
        x_idx = ts_map[e['timestamp']]
        if x_idx not in grouped_trades:
            grouped_trades[x_idx] = []
        grouped_trades[x_idx].append(e)

    next_up = True 

    for x_idx in sorted(grouped_trades.keys()):
        group = grouped_trades[x_idx]
        avg_price = sum(t["price"] for t in group) / len(group)
        
        if len(group) == 1:
            # SINGLE TRADE
            e = group[0]
            style_key = (e["type"], e["side"])
            if style_key in STYLES:
                color, marker, label = STYLES[style_key]
            else:
                color, marker, label = ("gray", "o", "Unknown")
            
            ax1.scatter(x_idx, e["price"], color=color, marker=marker,
                        s=60, linewidths=2.5 if marker=="x" else 1.0,
                        alpha=0.9, zorder=5)
            
            direction = 1 if next_up else -1
            next_up = not next_up
            candle_len = 15 * 0.7 # 30% smaller
            end_y = e["price"] + direction * candle_len
            
            ax1.vlines(x_idx, e["price"], end_y, colors=color, linewidth=1.5, alpha=0.6)
            
            # Format to 2 decimals for Shares and Cost
            label_text = f"{e['shares']:.2f}sh\n${e['cost']:.2f}"
            
            ax1.annotate(
                label_text,
                xy=(x_idx, end_y),
                xytext=(0, direction * 2),
                textcoords="offset points",
                ha="center", va="bottom" if direction > 0 else "top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
            )
            
        else:
            # MULTIPLE TRADES
            count = len(group)
            first_e = group[0]
            same_side = all((t["type"] == first_e["type"] and t["side"] == first_e["side"]) for t in group)
            
            if same_side:
                style_key = (first_e["type"], first_e["side"])
                color, _, _ = STYLES.get(style_key, ("gray", "o", ""))
            else:
                color = "#1f77b4" 

            ax1.scatter(x_idx, avg_price, color="white", marker="o", s=300, edgecolors=color, linewidth=2, zorder=5)
            ax1.text(x_idx, avg_price, str(count), ha="center", va="center", fontsize=9, fontweight="bold", color=color, zorder=6)
            
            direction = 1 if next_up else -1
            next_up = not next_up
            
            info_lines = []
            for idx, t in enumerate(group):
                if idx < 5:
                    # Format to 2 decimals for Shares and Cost
                    info_lines.append(f"{t['shares']:.2f}sh ${t['cost']:.2f} ({t['side']})")
                else:
                    remaining = len(group) - 5
                    info_lines.append(f"...+ {remaining} more")
                    break
            
            box_text = "\n".join(info_lines)
            
            # 30% SMALLER LINE logic
            raw_len = 25 + (len(info_lines) * 5)
            candle_len = raw_len * 0.7 
            
            end_y = avg_price + direction * candle_len
            
            ax1.vlines(x_idx, avg_price, end_y, colors=color, linewidth=2, alpha=0.6, linestyles="dotted")
            
            ax1.annotate(
                box_text,
                xy=(x_idx, end_y),       
                xytext=(0, direction * 2), 
                textcoords="offset points",
                ha="center", va="bottom" if direction > 0 else "top",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=color)
            )

    ax1.set_title(f"Trades for {market_title}")
    ax1.set_ylabel("Price (cents)")
    
    def time_formatter(x, pos):
        idx = int(x)
        if 0 <= idx < len(unique_timestamps):
            ts = float(unique_timestamps[idx])
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
            ms = int(round((ts % 1) * 1000))
            if ms != 0:
                return dt.strftime('%H:%M:%S') + f'.{ms:03d}Z'
            return dt.strftime('%H:%M:%S') + 'Z'
        return ""

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    ax1.set_yticks(range(0, 101, 10))
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    vol_yes_per_ts = [0.0] * len(unique_timestamps)
    vol_no_per_ts = [0.0] * len(unique_timestamps)
    
    for x_idx, group in grouped_trades.items():
        for t in group:
            if t["type"] == "Buy":
                if t["side"] == "Up":
                    vol_yes_per_ts[x_idx] += t["shares"]
                else:
                    vol_no_per_ts[x_idx] += t["shares"]

    x_range = np.arange(len(unique_timestamps))
    vol_ax = ax1.inset_axes([0, 0.0, 1.0, 0.2], sharex=ax1)
    vol_ax.patch.set_alpha(0)
    vol_ax.bar(x_range - 0.35/2, vol_yes_per_ts, width=0.35,
               color="green", alpha=0.18, label="Buy YES volume")
    vol_ax.bar(x_range + 0.35/2, vol_no_per_ts, width=0.35,
               color="red", alpha=0.18, label="Buy NO volume")
    vol_ax.set_yticks([])
    vol_ax.set_xticks([])
    vol_ax.set_xlim(-0.5, len(unique_timestamps) - 0.5)

    # ---------------------------------------------------
    # SECOND: CUMULATIVE BUY SHARES + COST
    # ---------------------------------------------------
    ax2.plot(x_indices, cum_yes, color="green", alpha=0.3, linewidth=1, label="Cum Buy YES (sh)")
    ax2.fill_between(x_indices, cum_yes, color="green", alpha=0.1)
    
    ax2.plot(x_indices, cum_no, color="red", alpha=0.3, linewidth=1, label="Cum Buy NO (sh)")
    ax2.fill_between(x_indices, cum_no, color="red", alpha=0.1)
    
    ax2.set_ylabel("Cumulative buy volume (sh)")
    ax2.grid(axis='y', alpha=0.2)
    ax2.set_xticks([])
    ax2.set_title("Cumulative Buys (shares + dollars)")

    max_cum = max(cum_yes.max() if len(cum_yes) else 0, cum_no.max() if len(cum_no) else 0)
    ax2.set_ylim(0, max_cum * 1.15 + 1e-6)

    ax2_cost = ax2.twinx()
    ax2_cost.plot(x_indices, cum_yes_cost, color="green", linewidth=1.8,
                  linestyle="--", alpha=0.7, label="Cumulative Buy YES ($)")
    ax2_cost.plot(x_indices, cum_no_cost, color="red", linewidth=1.8,
                  linestyle="--", alpha=0.7, label="Cumulative Buy NO ($)")
    ax2_cost.set_ylabel("Cumulative buy cost ($)", color="gray", fontsize=9)
    ax2_cost.tick_params(axis='y', labelsize=8, colors="gray")
    ax2_cost.spines['right'].set_alpha(0.3)
    handles2, labels2 = ax2_cost.get_legend_handles_labels()
    handles1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    cum_stats_text = (
        f"YES: {cum_yes_total:.2f} sh / $ {cum_yes_cost_total:.2f}\n"
        f"NO:  {cum_no_total:.2f} sh / $ {cum_no_cost_total:.2f}"
    )
    ax2.text(
        0.01, 0.02, cum_stats_text,
        transform=ax2.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray")
    )

    # ---------------------------------------------------
    # THIRD: DOLLAR EXPOSURE
    # ---------------------------------------------------
    ax3.grid(alpha=0.3)
    ax3.plot(x_indices, yes_curve, color="green", linewidth=2, label="YES Exposure ($)")
    ax3.plot(x_indices, no_curve, color="red", linewidth=2, label="NO Exposure ($)")
    ax3.plot(x_indices, net_curve, color="blue", linewidth=2, label="NET Exposure ($ total)")

    if len(yes_curve) > 0:
        yes_peak = int(np.argmax(yes_curve))
        no_peak = int(np.argmax(no_curve))
        ax3.annotate("YES peak $", (x_indices[yes_peak], yes_curve[yes_peak]), xytext=(0, -20),
                     textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color="green"), color="green")
        ax3.annotate("NO peak $", (x_indices[no_peak], no_curve[no_peak]), xytext=(0, -20),
                     textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color="red"), color="red")
        
        last_x = x_indices[-1]
        ax3.annotate(f"$ {yes_curve[-1]:.2f}", (last_x, yes_curve[-1]), xytext=(15, 0), textcoords="offset points", color="green")
        ax3.annotate(f"$ {no_curve[-1]:.2f}", (last_x, no_curve[-1]), xytext=(15, 0), textcoords="offset points", color="red")
        ax3.annotate(f"$ {net_curve[-1]:.2f}", (last_x, net_curve[-1]), xytext=(15, 0), textcoords="offset points", color="blue")

    ax3.set_title("Dollar Exposure")
    ax3.set_ylabel("Exposure ($)")
    ax3.set_xticks([])
    ax3.legend(loc="upper left")

    pnl_text = (
        f"MARKET RESOLVED: {resolved_side}\n\n"
        f"Remaining YES shares: {remaining_yes:.2f}\n"
        f"Remaining NO shares:  {remaining_no:.2f}\n\n"
        f"Final Value: $ {final_value:.2f}\n"
        f"Total Spent (net exposure): $ {total_spent:.2f}\n\n"
        f"FINAL PNL: $ {pnl:.2f}"
    )

    summary = (
        f"YES (Up)  Buy: {yes_buy_sh:.2f} sh ($ {yes_buy_cost:.2f}) "
        f" | Sell: {yes_sell_sh:.2f} sh ($ {yes_sell_cost:.2f})\n"
        f"NO  (Down) Buy: {no_buy_sh:.2f} sh ($ {no_buy_cost:.2f}) "
        f" | Sell: {no_sell_sh:.2f} sh ($ {no_sell_cost:.2f})"
    )
    fig.text(0.01, 0.01, summary, ha="left", va="bottom", fontsize=11,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"))
    fig.text(0.99, 0.06, pnl_text, ha="right", va="top", fontsize=12,
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="black"))


    # ---------------------------------------------------
    # BOTTOM: SHARES EXPOSURE
    # ---------------------------------------------------
    ax4.grid(alpha=0.3)
    ax4.plot(x_indices, yes_sh_curve, color="green", linewidth=2, label="YES Exposure (shares)")
    ax4.plot(x_indices, no_sh_curve, color="red", linewidth=2, label="NO Exposure (shares)")
    ax4.plot(x_indices, net_sh_curve, color="blue", linewidth=2, label="NET Exposure (shares)")

    if len(yes_sh_curve) > 0:
        yes_sh_peak = int(np.argmax(yes_sh_curve))
        no_sh_peak = int(np.argmax(no_sh_curve))
        ax4.annotate("YES peak sh", (x_indices[yes_sh_peak], yes_sh_curve[yes_sh_peak]), xytext=(0, -20),
                     textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color="green"), color="green")
        ax4.annotate("NO peak sh", (x_indices[no_sh_peak], no_sh_curve[no_sh_peak]), xytext=(0, -20),
                     textcoords="offset points", ha='center', arrowprops=dict(arrowstyle="->", color="red"), color="red")
        
        ax4.annotate(f"{yes_sh_curve[-1]:.2f} sh", (last_x, yes_sh_curve[-1]), xytext=(15, 0), textcoords="offset points", color="green")
        ax4.annotate(f"{no_sh_curve[-1]:.2f} sh", (last_x, no_sh_curve[-1]), xytext=(15, 0), textcoords="offset points", color="red")
        ax4.annotate(f"{net_sh_curve[-1]:.2f} sh", (last_x, net_sh_curve[-1]), xytext=(15, 0), textcoords="offset points", color="blue")

    ax4.set_title("Shares Exposure")
    ax4.set_ylabel("Shares")
    
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')
    
    ax4.legend(loc="upper left")

    # Sync X limits
    xlim_range = (-0.5, len(unique_timestamps) - 0.5)
    for axis in (ax1, ax2, ax3, ax4):
        axis.set_xlim(*xlim_range)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close('all')

# ---------------------------------------------------
# PROCESS USER DATA
# ---------------------------------------------------
def process_user_data(user_label, raw_data, market_title, resolved_side, output_dir):
    print(f"Processing data for {user_label}...")
    
    # Parse trades
    parsed = []
    
    for item in raw_data:
        entry = {}
        raw_side = item.get("side", "BUY").upper()
        entry["type"] = "Buy" if raw_side == "BUY" else "Sell"
        entry["market"] = item.get("title", "")
        entry["side"] = item.get("outcome", "Up") 
        entry["price"] = float(item.get("price", 0)) * 100.0  # Convert to cents
        entry["shares"] = float(item.get("size", 0))
        entry["cost"] = float(item.get("price", 0)) * entry["shares"]
        entry["timestamp"] = float(item.get("timestamp", 0))
        parsed.append(entry)

    if not parsed:
        print(f"No trades for {user_label}. Skipping.")
        return

    prices = [e["price"] for e in parsed]

    # Exposure Curves
    yes_curve = []
    no_curve = []
    net_curve = []

    yes_sh_curve = []
    no_sh_curve = []
    net_sh_curve = []

    yes_exp = no_exp = 0
    yes_sh_exp = no_sh_exp = 0

    for e in parsed:
        # Dollar exposure
        if e["side"] == "Up":   # YES
            yes_exp += e["cost"] if e["type"] == "Buy" else -e["cost"]
        else:                   # NO
            no_exp += e["cost"] if e["type"] == "Buy" else -e["cost"]

        # Shares exposure
        if e["side"] == "Up":
            yes_sh_exp += e["shares"] if e["type"] == "Buy" else -e["shares"]
        else:
            no_sh_exp += e["shares"] if e["type"] == "Buy" else -e["shares"]

        yes_curve.append(yes_exp)
        no_curve.append(no_exp)
        net_curve.append(yes_exp + no_exp)

        yes_sh_curve.append(yes_sh_exp)
        no_sh_curve.append(no_sh_exp)
        net_sh_curve.append(yes_sh_exp + no_sh_exp)


    # Final PnL Calc
    remaining_yes = yes_sh_curve[-1]
    remaining_no = no_sh_curve[-1]
    total_spent = net_curve[-1]

    if resolved_side == "YES":
        final_value = remaining_yes * 1.0
    else:
        final_value = remaining_no * 1.0

    pnl = final_value - total_spent

    # Global Totals
    yes_buy_sh = yes_buy_cost = 0
    yes_sell_sh = yes_sell_cost = 0
    no_buy_sh = no_buy_cost = 0
    no_sell_sh = no_sell_cost = 0

    # Arrays for cumulative plotting
    raw_vol_yes = []
    raw_vol_no = []
    raw_cost_yes = []
    raw_cost_no = []

    for e in parsed:
        is_yes = (e["side"] == "Up")
        is_buy = (e["type"] == "Buy")
        
        if is_buy:
            if is_yes:
                yes_buy_sh += e["shares"]
                yes_buy_cost += e["cost"]
                raw_vol_yes.append(e["shares"])
                raw_vol_no.append(0)
                raw_cost_yes.append(e["cost"])
                raw_cost_no.append(0)
            else:
                no_buy_sh += e["shares"]
                no_buy_cost += e["cost"]
                raw_vol_yes.append(0)
                raw_vol_no.append(e["shares"])
                raw_cost_yes.append(0)
                raw_cost_no.append(e["cost"])
        else:
            # Sell
            raw_vol_yes.append(0)
            raw_vol_no.append(0)
            raw_cost_yes.append(0)
            raw_cost_no.append(0)
            if is_yes:
                yes_sell_sh += e["shares"]
                yes_sell_cost += e["cost"]
            else:
                no_sell_sh += e["shares"]
                no_sell_cost += e["cost"]

    cum_yes = np.cumsum(raw_vol_yes)
    cum_no = np.cumsum(raw_vol_no)
    cum_yes_cost = np.cumsum(raw_cost_yes)
    cum_no_cost = np.cumsum(raw_cost_no)

    cum_yes_total = cum_yes[-1] if len(cum_yes) > 0 else 0
    cum_no_total = cum_no[-1] if len(cum_no) > 0 else 0
    cum_yes_cost_total = cum_yes_cost[-1] if len(cum_yes_cost) > 0 else 0
    cum_no_cost_total = cum_no_cost[-1] if len(cum_no_cost) > 0 else 0

    # Generate Chart
    chart_filename = f"chart-{user_label}.png"
    chart_path = os.path.join(output_dir, chart_filename)
    
    generate_chart(
        parsed, chart_path, market_title, resolved_side,
        yes_curve, no_curve, net_curve, 
        yes_sh_curve, no_sh_curve, net_sh_curve,
        yes_buy_sh, yes_buy_cost, yes_sell_sh, yes_sell_cost,
        no_buy_sh, no_buy_cost, no_sell_sh, no_sell_cost,
        pnl, remaining_yes, remaining_no, final_value, total_spent,
        cum_yes, cum_no, cum_yes_cost, cum_no_cost,
        cum_yes_total, cum_no_total, cum_yes_cost_total, cum_no_cost_total
    )
    print(f"Generated chart: {chart_path}")

    # Generate Report
    report_filename = f"report-{user_label}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    write_stats_report(
        report_path,
        market_title,
        resolved_side,
        len(parsed),
        remaining_yes,
        remaining_no,
        final_value,
        total_spent,
        pnl,
        yes_buy_sh,
        yes_buy_cost,
        yes_sell_sh,
        yes_sell_cost,
        no_buy_sh,
        no_buy_cost,
        no_sell_sh,
        no_sell_cost,
        cum_yes_total,
        cum_no_total,
        cum_yes_cost_total,
        cum_no_cost_total,
        yes_curve,
        no_curve,
        net_curve,
        yes_sh_curve,
        no_sh_curve,
        net_sh_curve,
        prices,
        parsed,
    )
    print(f"Generated report: {report_path}")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def _parse_cli_args():
    """
    Parse CLI arguments.

    Usage examples:
      - python get-history.py
          -> market name asked via input, resolved side inferred / prompted
      - python get-history.py \"Bitcoin Up or Down - February 16, 9:00AM-9:15AM ET\"
          -> market name from argv[1], resolved side inferred / prompted
      - python get-history.py YES \"Bitcoin Up or Down - February 16, 9:00AM-9:15AM ET\"
          -> resolved side forced to YES, market name from argv[2]
      - python get-history.py AUTO \"Bitcoin Up or Down - February 16, 9:00AM-9:15AM ET\"
          -> try to infer resolved side automatically, otherwise abort
    """
    market_arg = None
    resolved_arg = None
    search_mode = False

    # On veut pouvoir appeler:
    #   python get-history.py Bitcoin Up or Down - February 16, 9:00AM-9:15AM ET
    # sans guillemets, donc on reconstruit le nom du market
    if len(sys.argv) > 1:
        # Mode interactif explicite:
        #   python getHistory.py search
        if sys.argv[1].strip().lower() == "search":
            search_mode = True
            # Optionnel: autoriser un resolved side après "search"
            if len(sys.argv) > 2:
                resolved_arg = normalize_resolved_arg(sys.argv[2])
            return market_arg, resolved_arg, search_mode

        first = sys.argv[1]
        norm_first = normalize_resolved_arg(first)
        if norm_first:
            # Premier argument = resolved side (YES/NO/AUTO)
            resolved_arg = norm_first
            if len(sys.argv) > 2:
                # Tout le reste devient le nom du market
                market_arg = " ".join(sys.argv[2:])
        else:
            # Pas de resolved side explicite : tout ce qui suit est le nom du market
            market_arg = " ".join(sys.argv[1:])

    return market_arg, resolved_arg, search_mode


def main():
    # CLI can provide market name and/or resolved side
    market_arg, resolved_arg, search_mode = _parse_cli_args()

    # 1. Get Market Query
    if market_arg and not search_mode:
        market_query = market_arg.strip()
    else:
        market_query = input('Enter market name to search (ex: "Bitcoin Up or Down - March 23, 8:55AM-9:00AM ET"): ').strip()
        if not market_query:
            print("Market name is required.")
            return

    results = search_markets(market_query)
    if not results:
        print("No market found for that query.")
        return
    
    # Use the first one for single-run mode
    event, market = results[0]

    market_title = (
        market.get("question")
        or market.get("title")
        or event.get("title", "Unknown Market")
    )
    condition_id = market.get("conditionId") or ""
    print(f"Found market: {market_title}")
    print(f"Condition ID: {condition_id}")

    # 1.5. Wait if Market Hasn't Ended + 2 Minutes
    # In search mode we process immediately (no auto-wait).
    if not search_mode:
        end_time_utc = parse_market_end_time(market_title)
        if end_time_utc:
            # Calculate target time (End + 2 minutes)
            target_time = end_time_utc + datetime.timedelta(minutes=2)
            
            # Current time in UTC (timezone-aware)
            now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            
            wait_seconds = (target_time - now_utc).total_seconds()
            
            if wait_seconds > 0:
                print(f"\nMarket ends at {end_time_utc.strftime('%H:%M')} UTC. Waiting until {target_time.strftime('%H:%M')} UTC (+2 min).")
                print(f"Sleeping for {wait_seconds:.0f} seconds...")
                try:
                    time.sleep(wait_seconds)
                except KeyboardInterrupt:
                    print("\nWait interrupted by user. Exiting.")
                    return
                print("Resuming...")
        else:
            print("Could not parse end time from market name. Skipping auto-wait.")

    # 2. Create Directory
    safe_name = sanitize_filename(market_title)
    if len(safe_name) > 50:
        safe_name = safe_name[:50]
    output_dir = os.path.join("Datas", "report", safe_name.strip())
    if os.path.isdir(output_dir):
        print(f"Output directory already exists, overwriting: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)
    print(f"Output directory: {abs_output_dir}")

    # 3. Fetch Trades for all Users
    all_users_trades = {}
    combined_trades = []

    for label, address in USERS.items():
        print(f"Fetching trades for {label} ({address})...")
        trades = fetch_trades(condition_id, address)
        all_users_trades[label] = trades
        combined_trades.extend(trades)

    if not combined_trades:
        print("No trades found for any user.")
        return

    # 4. Infer Resolved Side (Global)
    resolved_side = None
    if resolved_arg in {"YES", "NO"}:
        resolved_side = resolved_arg
    else:
        inferred, latest = infer_resolved_side_from_trades(combined_trades)
        if inferred:
            resolved_side = inferred
            price = float(latest.get("price", 0))
            outcome = latest.get("outcome", "")
            ts = latest.get("timestamp", 0)
            print(f"Inferred resolved side from combined data: {resolved_side} (latest trade {outcome} @ {price}, ts {ts})")
        else:
            if resolved_arg == "AUTO":
                print("Could not infer resolved side automatically.")
                return
            resolved_side = prompt_resolved_side(None)
            if not resolved_side:
                print("Resolved side is required.")
                return

    # 5. Process and Generate per User (gabagool, perso)
    for label, trades in all_users_trades.items():
        if not trades:
            continue
        # Sort trades by timestamp
        trades.sort(key=lambda x: x.get("timestamp", 0))
        process_user_data(label, trades, market_title, resolved_side, output_dir)

    print("Done.")
    print(f"OUTPUT_DIR={abs_output_dir}")


def process_market_safely(market, event, resolved_side='AUTO'):
    """
    Encapsulates the logic to process a single market (fetch trades, generate report/chart).
    Returns True if successful, False otherwise.
    """
    try:
        market_title = (
            market.get("question")
            or market.get("title")
            or event.get("title", "Unknown Market")
        )
        condition_id = market.get("conditionId") or ""
        print(f"\n[PROCESSING] {market_title} (ID: {condition_id})")
        
        # Build output dir name
        safe_name = sanitize_filename(market_title)
        if len(safe_name) > 50:
            safe_name = safe_name[:50]
        output_dir = os.path.join("Datas", "report", safe_name.strip())

        # Skip if already processed (folder exists on disk)
        if os.path.isdir(output_dir):
            print(f"  Already processed (folder exists). Skipping.")
            return True
        
        # Fetch Trades for all Users
        all_users_trades = {}
        combined_trades = []

        for label, address in USERS.items():
            trades = fetch_trades(condition_id, address)
            all_users_trades[label] = trades
            combined_trades.extend(trades)

        if not combined_trades:
            print(f"  No trades found for {market_title}. Skipping.")
            return True
        
        # Create directory only if we have trades
        os.makedirs(output_dir, exist_ok=True)

        # Infer Resolved Side
        final_resolved_side = None
        if resolved_side in {"YES", "NO"}:
            final_resolved_side = resolved_side
        else:
            inferred, latest = infer_resolved_side_from_trades(combined_trades)
            if inferred:
                final_resolved_side = inferred
                print(f"  Inferred resolution: {final_resolved_side}")
            else:
                print("  Could not infer resolved side. Defaulting to UNKNOWN for report.")
                final_resolved_side = "UNKNOWN"

        # Process per User
        for label, trades in all_users_trades.items():
            if not trades:
                continue
            trades.sort(key=lambda x: x.get("timestamp", 0))
            process_user_data(label, trades, market_title, final_resolved_side, output_dir)

        # Demo fills REMOVED as requested
            
        print(f"[SUCCESS] Processed {market_title}")
        return True

    except Exception as e:
        print(f"[ERROR] processing {market.get('question')}: {e}")
        return False


def _format_hour_ampm(hour24):
    """Convert 24h hour to ('5', 'PM') style tuple."""
    if hour24 == 0:
        return "12", "AM"
    elif hour24 < 12:
        return str(hour24), "AM"
    elif hour24 == 12:
        return "12", "PM"
    else:
        return str(hour24 - 12), "PM"


def generate_market_names_to_check(now_utc):
    """
    Given current UTC time, generate the LAST closed market name
    for each timeframe (M5, M15, H1) per crypto.
    Only returns markets that closed at least 2 minutes ago.
    Returns list of (market_name, timeframe) tuples.
    """
    # Convert UTC to ET (EST = UTC-5)
    now_et = now_utc - datetime.timedelta(hours=5)
    
    names = []
    cryptos = ["Bitcoin", "Ethereum", "Solana", "XRP"]
    
    for crypto in cryptos:
        # --- M5: find the last 5-min boundary that is >= 2 mins ago ---
        # Round down current ET minute to nearest 5
        m5_end_minute = (now_et.minute // 5) * 5
        m5_end = now_et.replace(minute=m5_end_minute, second=0, microsecond=0)
        # If that boundary is less than 2 mins ago, go back one more period
        if (now_et - m5_end).total_seconds() < 5:
            m5_end = m5_end - datetime.timedelta(minutes=5)
        m5_start = m5_end - datetime.timedelta(minutes=5)
        
        s_h, s_ampm = _format_hour_ampm(m5_start.hour)
        e_h, e_ampm = _format_hour_ampm(m5_end.hour)
        s_str = f"{s_h}:{m5_start.minute:02d}{s_ampm}"
        e_str = f"{e_h}:{m5_end.minute:02d}{e_ampm}"
        name = f"{crypto} Up or Down - {m5_end.strftime('%B')} {m5_end.day}, {s_str}-{e_str} ET"
        names.append((name, "M5"))
        
        # --- M15: find the last 15-min boundary >= 2 mins ago ---
        m15_end_minute = (now_et.minute // 15) * 15
        m15_end = now_et.replace(minute=m15_end_minute, second=0, microsecond=0)
        if (now_et - m15_end).total_seconds() < 5:
            m15_end = m15_end - datetime.timedelta(minutes=15)
        m15_start = m15_end - datetime.timedelta(minutes=15)
        
        s_h, s_ampm = _format_hour_ampm(m15_start.hour)
        e_h, e_ampm = _format_hour_ampm(m15_end.hour)
        s_str = f"{s_h}:{m15_start.minute:02d}{s_ampm}"
        e_str = f"{e_h}:{m15_end.minute:02d}{e_ampm}"
        name = f"{crypto} Up or Down - {m15_end.strftime('%B')} {m15_end.day}, {s_str}-{e_str} ET"
        names.append((name, "M15"))
        
        # --- H1: find the last hour boundary >= 2 mins ago ---
        h1_end = now_et.replace(minute=0, second=0, microsecond=0)
        if (now_et - h1_end).total_seconds() < 5:
            h1_end = h1_end - datetime.timedelta(hours=1)
        h1_start = h1_end - datetime.timedelta(hours=1)
        
        s_h, s_ampm = _format_hour_ampm(h1_start.hour)
        name = f"{crypto} Up or Down - {h1_start.strftime('%B')} {h1_start.day}, {s_h}{s_ampm} ET"
        names.append((name, "H1"))
    
    return names


def search_market_by_name(market_name):
    """Search for a specific market by exact name. Returns (event, market) or (None, None)."""
    try:
        resp = requests.get(SEARCH_URL, params={"q": market_name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  Error searching '{market_name}': {exc}")
        return None, None

    events = data.get("events", []) if isinstance(data, dict) else []
    for event in events:
        for m in event.get("markets", []):
            title = (m.get("question") or "").strip()
            if title.lower() == market_name.lower():
                return event, m
    return None, None


def run_continuous_loop():
    """
    Continuously monitor for BTC, ETH, SOL and XRP Up/Down markets (M5, M15, H1).
    Uses deterministic market name generation based on current ET time
    to reliably find recently closed contracts.
    
    Sleeps precisely until the next market check time (XX:00:30, XX:05:30, etc).
    """
    print("Starting continuous loop monitor...")
    print("Press Ctrl+C to stop.")
    
    processed_names = set()
    
    while True:
        try:
            now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            now_et = now_utc - datetime.timedelta(hours=5)
            
            # 1. Generate candidates for the current time
            print(f"\n--- Scan at {now_utc.strftime('%H:%M:%S')} UTC ({now_et.strftime('%H:%M:%S')} ET) ---")
            
            candidates = generate_market_names_to_check(now_utc)
            
            for market_name, tf in candidates:
                if market_name in processed_names:
                    continue
                
                # Check / Process
                event, market = search_market_by_name(market_name)
                
                if market:
                    # cid = market.get("conditionId", "")
                    print(f"[{tf}] Found: {market_name}")
                    success = process_market_safely(market, event, resolved_side='AUTO')
                    if success:
                        processed_names.add(market_name)
            
            # 2. Smart Sleep Calculation
            # We want to wake up at the next XX:00:30, XX:05:30, ...
            # Get current time again after processing
            now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            
            # Find next 5-minute mark
            # Example: 12:02:10 -> next is 12:05:00
            next_minute_mark = (now_utc.minute // 5 + 1) * 5
            next_wake_time = now_utc.replace(minute=0, second=30, microsecond=0) + datetime.timedelta(minutes=next_minute_mark)
            
            # If next_wake_time is already passed (rare/edge case if processing took long), add 5 mins
            if next_wake_time <= now_utc:
                next_wake_time += datetime.timedelta(minutes=5)
            
            sleep_seconds = (next_wake_time - now_utc).total_seconds()
            
            print(f"Sleeping {sleep_seconds:.1f}s until {next_wake_time.strftime('%H:%M:%S')} UTC...")
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            print("\nStopping loop.")
            break
        except Exception as e:
            print(f"Error in loop: {e}")
            # If crash, sleep 60s to avoid spam
            time.sleep(60)


if __name__ == "__main__":
    # If no args or specific flag, run loop?
    # User said: "I have to run it once, and then it continues..."
    # So default behavior should probably be loop if no specific market name provided.
    
    if len(sys.argv) == 1:
        run_continuous_loop()
    elif len(sys.argv) > 1 and sys.argv[1].upper() == "--LOOP":
        run_continuous_loop()
    else:
        main()
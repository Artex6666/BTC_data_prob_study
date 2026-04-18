"""Filtre par jour UTC partagé par les scripts gen_custom_chart*.py."""
from __future__ import annotations

import pandas as pd


def utc_day_bounds(day_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Jour calendaire UTC [start, end) pour une date type YYYY-MM-DD.
    Les timestamps naïfs dans les contrats sont traités comme UTC.
    """
    d = pd.Timestamp(day_str).normalize()
    if d.tzinfo is None:
        start = d.tz_localize("UTC")
    else:
        start = d.tz_convert("UTC").normalize()
    end = start + pd.Timedelta(days=1)
    return start, end


def filter_contracts_by_utc_day(cts, start: pd.Timestamp, end: pd.Timestamp):
    """Garde les contrats dont ce (clôture) tombe dans [start, end) UTC."""
    if not cts:
        return []
    out = []
    for c in cts:
        ce = pd.Timestamp(c["ce"])
        if ce.tzinfo is None:
            ce = ce.tz_localize("UTC")
        else:
            ce = ce.tz_convert("UTC")
        if start <= ce < end:
            out.append(c)
    return out

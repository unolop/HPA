#!/usr/bin/env python3
"""
Poll OpenAI org usage/cost endpoints and print a live terminal summary.

Env:
  OPENAI_API_KEY      (required)
  OPENAI_ORG_ID       (optional)
  OPENAI_PROJECT_ID   (optional)

Examples:
  python scripts/monitor_openai_usage.py
  python scripts/monitor_openai_usage.py --interval 120 --window-days 1
  python scripts/monitor_openai_usage.py --once
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple


API_BASE = "https://api.openai.com/v1"


def _headers(api_key: str) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    org = os.getenv("OPENAI_ORG_ID")
    project = os.getenv("OPENAI_PROJECT_ID")
    if org:
        h["OpenAI-Organization"] = org
    if project:
        h["OpenAI-Project"] = project
    return h


def _http_get(url: str, api_key: str, timeout: int = 20) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET", headers=_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}") from e


def _sum_field(obj: Any, key: str) -> float:
    total = 0.0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)):
                total += float(v)
            else:
                total += _sum_field(v, key)
    elif isinstance(obj, list):
        for x in obj:
            total += _sum_field(x, key)
    return total


def _first_number(obj: Any, keys: Tuple[str, ...]) -> Optional[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (int, float)):
                return float(v)
            got = _first_number(v, keys)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for x in obj:
            got = _first_number(x, keys)
            if got is not None:
                return got
    return None


def _build_usage_url(start: int, end: int) -> str:
    params = urllib.parse.urlencode(
        {"start_time": start, "end_time": end, "bucket_width": "1d"}
    )
    return f"{API_BASE}/organization/usage/completions?{params}"


def _build_cost_url(start: int, end: int) -> str:
    params = urllib.parse.urlencode({"start_time": start, "end_time": end})
    return f"{API_BASE}/organization/costs?{params}"


def fetch_snapshot(api_key: str, window_days: int) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    start = int((now - dt.timedelta(days=window_days)).timestamp())
    end = int(now.timestamp())

    usage = _http_get(_build_usage_url(start, end), api_key)
    costs = _http_get(_build_cost_url(start, end), api_key)

    output_tokens = _sum_field(usage, "output_tokens")
    input_tokens = _sum_field(usage, "input_tokens")
    req_count = _sum_field(usage, "num_model_requests")

    # Handle common cost field variants.
    usd = (
        _first_number(costs, ("amount", "total", "amount_usd"))
        or _sum_field(costs, "amount")
        or 0.0
    )
    # Some responses return cents-like integer.
    if usd > 10000:
        usd = usd / 100.0

    return {
        "ts": now,
        "window_days": window_days,
        "requests": int(req_count),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": float(usd),
    }


def print_snapshot(s: Dict[str, Any]) -> None:
    ts = s["ts"].astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(
        f"[{ts}] last {s['window_days']}d | "
        f"req={s['requests']:,} "
        f"in_tok={s['input_tokens']:,} "
        f"out_tok={s['output_tokens']:,} "
        f"cost_usd=${s['usd']:.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    p.add_argument("--window-days", type=int, default=30, help="Rolling window in days.")
    p.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    while True:
        try:
            s = fetch_snapshot(api_key=api_key, window_days=args.window_days)
            print_snapshot(s)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

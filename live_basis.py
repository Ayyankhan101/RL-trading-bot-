"""
Live paper trading for the funding-rate carry strategy.

    python live_basis.py --once            # process settled intervals and exit
    python live_basis.py --poll 3600       # keep running (funding settles every 8h)
    python live_basis.py --status          # print the saved account
    python live_basis.py --reset --once    # start a fresh paper account

Delta neutral: long spot, short perpetual, equal notional. No orders are placed.
"""

import argparse
import json
import os
import time

import yaml

from live.basis_trader import LiveBasisTrader


def print_status(status: dict) -> None:
    weekly = status.get('weekly') or {}
    rows = [
        ("Strategy", f"{status['strategy']} ({status['mode']}, {status['leverage']:.0f}x)"),
        ("Equity", f"${status['equity']:,.2f}"),
        ("Total return", f"{status['total_return'] * 100:+.3f}%"),
        ("Position", "long spot / short perp" if status['in_position'] else "flat"),
        ("Funding collected", f"${status['funding_collected']:,.2f}"),
        ("Basis P&L", f"${status['basis_pnl']:,.2f}"),
        ("Fees paid", f"${status['costs_paid']:,.2f}"),
        ("Last funding rate", f"{status['last_funding_rate'] * 100:+.5f}% / 8h"),
        ("Annualized at that rate", f"{status['annualized_funding'] * 100:+.2f}%"),
        ("Trailing funding", f"{status['trailing_funding'] * 100:+.5f}% / 8h"),
        ("Current basis", f"{status['current_basis'] * 100:+.4f}%"),
        ("Intervals held", f"{status['intervals_in_position']}/{status['intervals_seen']}"),
        ("Liquidated by", f"a {status['liquidation_move'] * 100:.1f}% adverse move"),
        ("Liquidations", str(status['liquidations'])),
    ]
    if weekly:
        rows += [
            ("Positive weeks",
             f"{weekly['positive_weeks']}/{weekly['weeks']} "
             f"({weekly['positive_week_rate'] * 100:.1f}%)"),
            ("Mean weekly return", f"{weekly['mean_weekly_return'] * 100:+.4f}%"),
            ("Worst week", f"{weekly['worst_week'] * 100:+.4f}%"),
        ]

    print("\nFUNDING CARRY - LIVE PAPER ACCOUNT")
    print("-" * 62)
    for label, value in rows:
        print(f"{label:<26}{value:>36}")
    print("-" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--state-dir', default='results/live_basis')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--poll', type=int, default=3600)
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.status:
        path = os.path.join(args.state_dir, 'status.json')
        if not os.path.exists(path):
            raise SystemExit(f"No status at {path}. Run: python live_basis.py --once")
        with open(path) as file:
            print_status(json.load(file))
        return

    if args.reset:
        for name in ('account.json', 'equity.csv', 'status.json'):
            path = os.path.join(args.state_dir, name)
            if os.path.exists(path):
                os.remove(path)
        print(f"Reset paper account in {args.state_dir}")

    trader = LiveBasisTrader(config, state_dir=args.state_dir, symbol=args.symbol)

    if args.once or args.cycles == 1:
        print_status(trader.tick())
        return

    cycle = 0
    print(f"Polling every {args.poll}s. Funding settles every 8h. Ctrl-C to stop.")
    while args.cycles is None or cycle < args.cycles:
        cycle += 1
        try:
            status = trader.tick()
            if status['new_intervals']:
                print(f"[carry] cycle {cycle}: {status['new_intervals']} new interval(s), "
                      f"equity ${status['equity']:,.2f} "
                      f"({status['total_return'] * 100:+.3f}%)")
        except Exception as exc:                       # noqa: BLE001
            print(f"[carry] cycle {cycle} failed: {exc}")
        if args.cycles is not None and cycle >= args.cycles:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

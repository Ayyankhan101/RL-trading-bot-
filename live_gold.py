"""
Live paper trading for gold (COMEX GC=F), CFD cost model.

    python live_gold.py --once                 # process closed bars and exit
    python live_gold.py --catch-up 2000 --once # replay recent bars first
    python live_gold.py --poll 900             # run continuously (1h bars)
    python live_gold.py --status               # print the paper account
    python live_gold.py --reset --once         # fresh account

No orders are placed and no broker keys are needed. Fills are simulated at the
broker's quote: spread as a price offset plus per-lot commission.

This is the piece that turns a backtest into evidence. Every bar that arrives
after today is data no model was fitted to - and with a Deflated Sharpe of 0.54
against a 1.49 threshold, a forward record is the only thing that can settle
whether the gold edge is real.
"""

import argparse
import json
import os
import time

import yaml

from live.gold_trader import GoldLiveTrader
from strategies.account_sizing import account_table, format_account_table


def _bar_age(status: dict) -> str:
    """
    How long since the last processed bar.

    The silent failure mode is a process that is alive but has not advanced in
    days - a changed Yahoo response, a dead symbol. Uptime does not show it;
    this does. Gold closes at weekends, so nothing is flagged until ~65 hours,
    or every Monday morning would look like an outage.
    """
    import pandas as pd

    last = status.get('last_bar')
    if not last:
        return 'no bars processed yet'

    age_hours = (pd.Timestamp.now() - pd.Timestamp(last)).total_seconds() / 3600
    flag = '  <-- STALE' if age_hours > 65 else ''
    return f"{age_hours:.1f}h{flag}"


def print_status(status: dict) -> None:
    learning = status.get('online_learning', {})
    per_bar = learning.get('per_bar') or {}
    event = status.get('next_event')

    rows = [
        ("Instrument", f"{status.get('instrument', 'gold')} {status['symbol']} "
                       f"{status['interval']}"),
        ("Mode", f"{status['mode']} ({status['model_origin']} model)"),
        ("Last bar", f"{status['last_bar']}  ${status['last_price']:,.2f}"),
        ("Equity", f"${status['equity']:,.2f}"),
        ("Total return", f"{status['total_return'] * 100:+.3f}%"),
        ("Position", f"{status['btc_held']:.4f} oz (${status['position_value']:,.2f})"
                     if status['btc_held'] else "flat"),
        ("Drawdown", f"{status['drawdown'] * 100:.2f}%"),
        ("Round trips", f"{status['round_trips']} "
                        f"(win rate {status['win_rate'] * 100:.1f}%)"),
        ("Session gaps in data", str(status.get('session_gaps', 0))),
        ("Event blackouts", f"{status.get('blackout_windows', 0)} windows, "
                            f"{status.get('blocked_entries', 0)} entries suppressed"),
        ("Next high-impact event",
         f"{event['country']} {event['title'][:30]} at {event['starts'][:16]}"
         if event else "none scheduled"),
        ("Learning", f"{per_bar.get('gradient_updates', 0)} updates, "
                     f"{learning.get('promotions', 0)} promotions"),
        ("Bar age", _bar_age(status)),
    ]

    print("\nGOLD - LIVE PAPER ACCOUNT")
    print("-" * 72)
    for label, value in rows:
        print(f"{label:<26}{value:>46}")
    print("-" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--state-dir', default='results/live_gold')
    parser.add_argument('--symbol', default='GC=F')
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--model', help='Seed checkpoint')
    parser.add_argument('--poll', type=int, default=900)
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--catch-up', type=int)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--equity', type=float, default=20.0,
                        help='Account size, for the lot-sizing banner')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.status:
        path = os.path.join(args.state_dir, 'status.json')
        if not os.path.exists(path):
            raise SystemExit(f"No status at {path}. Run: python live_gold.py --once")
        with open(path) as file:
            print_status(json.load(file))
        return

    if args.reset:
        for name in ('account.json', 'equity.csv', 'trades.csv', 'decisions.csv',
                     'status.json', 'promotions.json', 'live_model.pt',
                     'bootstrap_log.csv'):
            path = os.path.join(args.state_dir, name)
            if os.path.exists(path):
                os.remove(path)
        print(f"Reset paper account in {args.state_dir}")

    trader = GoldLiveTrader(config=config, model_path=args.model,
                            state_dir=args.state_dir, symbol=args.symbol,
                            interval=args.interval)

    if args.once or args.cycles == 1:
        status = trader.tick(max_bars=args.catch_up)
        print_status(status)

        # What this account could actually hold, every run, so the leverage is
        # never out of sight.
        print()
        print(format_account_table(
            account_table(args.equity, status['last_price'], 0.0017, lots=(0.01,)),
            args.equity))
        return

    print(f"Polling every {args.poll}s. Ctrl-C to stop.")
    trader.run(poll_seconds=args.poll, max_cycles=args.cycles)


if __name__ == "__main__":
    main()

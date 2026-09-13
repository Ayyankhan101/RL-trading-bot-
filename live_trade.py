"""
Live paper trading for the Bitcoin RL bot.

Trades a simulated account against real, newly-closed exchange bars, and
periodically fine-tunes the model behind a promotion gate.

    python live_trade.py --once                    # process new bars and exit
    python live_trade.py --catch-up 500 --once     # replay the last 500 bars first
    python live_trade.py --poll 300                # run continuously
    python live_trade.py --status                  # print the current account

This places no orders and needs no exchange keys. It simulates fills at the
close of each bar, charging the same fee and slippage as the backtest.
"""

import argparse
import json
import os
from typing import Any, Dict

import yaml

from live.trader import LiveTrader


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as file:
        return yaml.safe_load(file)


def print_status(status: Dict[str, Any]) -> None:
    online = status.get('online_learning', {})
    benchmark = status.get('buy_hold_return_since_start')

    rows = [
        ("Mode", f"{status['mode']} ({status['model_origin']} model)"),
        ("Symbol / interval", f"{status['symbol']} {status['interval']} via {status['source']}"),
        ("Last bar", f"{status['last_bar']}  ${status['last_price']:,.2f}"),
        ("Equity", f"${status['equity']:,.2f}"),
        ("Total return", f"{status['total_return'] * 100:+.2f}%"),
        ("Buy & hold, same window",
         f"{benchmark * 100:+.2f}%" if benchmark is not None else "n/a"),
        ("Cash", f"${status['balance']:,.2f}"),
        ("Position", f"{status['btc_held']:.6f} BTC (${status['position_value']:,.2f})"),
        ("Unrealized", f"{status['unrealized_pct'] * 100:+.2f}%"
         if status['btc_held'] else "flat"),
        ("Drawdown", f"{status['drawdown'] * 100:.2f}%"),
        ("Round trips", f"{status['round_trips']} (win rate {status['win_rate'] * 100:.1f}%)"),
        ("Halted", str(status['halted'])),
        ("Online learning",
         f"{online.get('cycles', 0)} cycles, {online.get('promotions', 0)} promotions, "
         f"{online.get('bars_since_retrain', 0)}/{online.get('retrain_every_bars', 0)} bars to next"),
    ]

    print("\nLIVE PAPER ACCOUNT")
    print("-" * 58)
    for label, value in rows:
        print(f"{label:<24}{value:>34}")
    print("-" * 58)


def main() -> None:
    parser = argparse.ArgumentParser(description='Live paper trading')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', help='Seed checkpoint (used until the live model exists)')
    parser.add_argument('--state-dir', help='Override live.state_dir')
    parser.add_argument('--source', help='auto | binance | kraken | coinbase')
    parser.add_argument('--interval', help='1h | 4h | 1d')
    parser.add_argument('--poll', type=int, help='Seconds between cycles')
    parser.add_argument('--cycles', type=int, help='Stop after this many cycles')
    parser.add_argument('--once', action='store_true', help='Run a single cycle and exit')
    parser.add_argument('--catch-up', type=int,
                        help='On a fresh account, replay at most this many recent bars')
    parser.add_argument('--status', action='store_true', help='Print the saved status and exit')
    parser.add_argument('--reset', action='store_true', help='Delete the paper account first')
    args = parser.parse_args()

    config = load_config(args.config)
    live_config = config.get('live', {})

    state_dir = args.state_dir or live_config.get('state_dir', 'results/live')
    interval = args.interval or live_config.get('interval', '4h')
    source = args.source or live_config.get('source', 'auto')
    poll = args.poll or int(live_config.get('poll_seconds', 300))

    if args.status:
        status_path = os.path.join(state_dir, 'status.json')
        if not os.path.exists(status_path):
            raise SystemExit(f"No status at {status_path}. Run: python live_trade.py --once")
        with open(status_path) as file:
            print_status(json.load(file))
        return

    if args.reset:
        # The model is part of the account: leaving it behind would make a
        # "fresh" run silently start from a model trained during the last one.
        for name in ('account.json', 'equity.csv', 'trades.csv',
                     'decisions.csv', 'status.json', 'promotions.json',
                     'live_model.pt', 'bootstrap_log.csv'):
            path = os.path.join(state_dir, name)
            if os.path.exists(path):
                os.remove(path)
        print(f"Reset paper account in {state_dir}")

    trader = LiveTrader(
        config=config,
        model_path=args.model,
        state_dir=state_dir,
        symbol=live_config.get('symbol', 'BTCUSD'),
        interval=interval,
        source=source,
    )

    if args.once or args.cycles == 1:
        status = trader.tick(max_bars=args.catch_up)
        print_status(status)
        return

    print(f"Polling every {poll}s. Ctrl-C to stop.")
    trader.run(poll_seconds=poll, max_cycles=args.cycles)


if __name__ == "__main__":
    main()

import { useEffect, useState } from 'react';
import DrawdownChart from './components/DrawdownChart';
import EquityChart from './components/EquityChart';
import FoldTable from './components/FoldTable';
import CarryPanel from './components/CarryPanel';
import LivePanel from './components/LivePanel';
import MetricsTable from './components/MetricsTable';
import SummaryTiles from './components/SummaryTiles';
import TradeDistribution from './components/TradeDistribution';
import { loadCarry } from './lib/loadCarry';
import { loadLive } from './lib/loadLive';
import { loadResults } from './lib/loadResults';
import { pct, shortDate } from './lib/format';
import type { CarryBundle, LiveBundle, ResultsBundle } from './types';

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: ResultsBundle };

const LIVE_REFRESH_MS = 60_000;

export default function App() {
  const [state, setState] = useState<State>({ status: 'loading' });
  const [live, setLive] = useState<LiveBundle | null>(null);
  const [carry, setCarry] = useState<CarryBundle | null>(null);

  useEffect(() => {
    loadResults()
      .then((data) => setState({ status: 'ready', data }))
      .catch((error: Error) => setState({ status: 'error', message: error.message }));
  }, []);

  // The live account keeps moving while the page is open, so poll its
  // artifacts rather than showing whatever was on disk at first paint.
  useEffect(() => {
    let cancelled = false;

    const refresh = () => {
      loadLive()
        .then((bundle) => {
          if (!cancelled) setLive(bundle);
        })
        .catch(() => {
          if (!cancelled) setLive(null);
        });
      loadCarry()
        .then((bundle) => {
          if (!cancelled) setCarry(bundle);
        })
        .catch(() => {
          if (!cancelled) setCarry(null);
        });
    };

    refresh();
    const timer = window.setInterval(refresh, LIVE_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  if (state.status === 'loading') {
    return <div className="status">Loading results…</div>;
  }

  // No artifacts means no results - the page says so rather than showing
  // placeholder numbers that look like a finished run.
  if (state.status === 'error') {
    return (
      <div className="status">
        <p>No backtest artifacts found ({state.message}).</p>
        <pre>python backtest.py --data data/BTC.csv{'\n'}npm run dev</pre>
      </div>
    );
  }

  const { metrics, meta, equity, trades } = state.data;

  return (
    <div className="app">
      <header>
        <h1>Bitcoin RL Trading Bot</h1>
        <p>
          Live paper account plus walk-forward backtest results —{' '}
          {shortDate(meta.oos_start)} to {shortDate(meta.oos_end)}.
        </p>
        <div className="meta-line">
          <span>variant <code>{meta.variant}</code></span>
          <span>seed <code>{meta.seed ?? 'n/a'}</code></span>
          <span>fees <code>{pct(meta.transaction_cost, 2)}</code></span>
          <span>slippage <code>{pct(meta.slippage, 3)}</code></span>
          <span>bars/year <code>{Math.round(meta.periods_per_year)}</code></span>
          {meta.git_sha && <span>commit <code>{meta.git_sha.slice(0, 8)}</code></span>}
          <span>generated <code>{meta.generated_at}</code></span>
        </div>
      </header>

      <CarryPanel carry={carry} />

      <LivePanel live={live} />

      <SummaryTiles agent={metrics.rl_agent} />
      <EquityChart equity={equity} />
      <DrawdownChart equity={equity} />
      <MetricsTable metrics={metrics} />
      <FoldTable folds={meta.folds} />
      <TradeDistribution trades={trades} />

      <footer>
        Every figure on this page is read from <code>results/</code>, written by{' '}
        <code>backtest.py</code>. Nothing here is hardcoded or simulated. Past performance does
        not predict future results.
      </footer>
    </div>
  );
}

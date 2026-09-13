import { num, parseCsv } from './csv';
import type { EquityPoint, MetricsFile, ResultsBundle, RoundTrip, RunMeta } from '../types';

const BASE = 'results';

async function fetchText(name: string): Promise<string> {
  const response = await fetch(`${BASE}/${name}`);
  if (!response.ok) {
    throw new Error(`Could not read ${name} (HTTP ${response.status})`);
  }
  return response.text();
}

async function fetchJson<T>(name: string): Promise<T> {
  return JSON.parse(await fetchText(name)) as T;
}

function toEquity(rows: Record<string, string>[]): EquityPoint[] {
  return rows
    .map((row) => {
      const agent = num(row.rl_agent);
      if (agent === null) return null;
      return {
        timestamp: row.timestamp,
        time: new Date(row.timestamp).getTime(),
        rl_agent: agent,
        buy_and_hold: num(row.buy_and_hold),
        rsi_strategy: num(row.rsi_strategy),
        mean_reversion: num(row.mean_reversion),
      } satisfies EquityPoint;
    })
    .filter((point): point is EquityPoint => point !== null);
}

function toTrades(rows: Record<string, string>[]): RoundTrip[] {
  return rows
    .filter((row) => row.exit_timestamp)
    .map((row) => ({
      entry_timestamp: row.entry_timestamp,
      exit_timestamp: row.exit_timestamp,
      entry_price: num(row.entry_price) ?? 0,
      exit_price: num(row.exit_price) ?? 0,
      amount: num(row.amount) ?? 0,
      pnl: num(row.pnl) ?? 0,
      return_pct: num(row.return_pct) ?? 0,
      bars_held: num(row.bars_held) ?? 0,
      exit_reason: row.exit_reason ?? 'signal',
    }));
}

/** Load every artifact. Rejects rather than substituting defaults. */
export async function loadResults(): Promise<ResultsBundle> {
  const [metrics, meta, equityCsv, tradesCsv] = await Promise.all([
    fetchJson<MetricsFile>('metrics.json'),
    fetchJson<RunMeta>('run_meta.json'),
    fetchText('equity_curve.csv'),
    fetchText('trades.csv'),
  ]);

  return {
    metrics,
    meta,
    equity: toEquity(parseCsv(equityCsv)),
    trades: toTrades(parseCsv(tradesCsv)),
  };
}

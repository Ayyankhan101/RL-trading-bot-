import type { MetricsFile, StrategyKey } from '../types';
import { count, money, pct, ratio } from '../lib/format';

interface Props {
  metrics: MetricsFile;
}

const ROW_LABELS: Record<StrategyKey, string> = {
  rl_agent: 'RL agent',
  buy_and_hold: 'Buy & hold',
  rsi_strategy: 'RSI strategy',
  mean_reversion: 'Mean reversion (rule)',
};

const ORDER: StrategyKey[] = ['rl_agent', 'mean_reversion', 'buy_and_hold', 'rsi_strategy'];

function signed(value: number | null | undefined, formatted: string) {
  if (value === null || value === undefined || !Number.isFinite(value)) return <span>{formatted}</span>;
  return <span className={value >= 0 ? 'positive' : 'negative'}>{formatted}</span>;
}

/** The comparison table, rendered straight from metrics.json. */
export default function MetricsTable({ metrics }: Props) {
  const rows = ORDER.filter((key) => metrics[key]);

  return (
    <div className="panel">
      <h2>Strategy comparison</h2>
      <p className="hint">
        Every cell is read from <code>results/metrics.json</code>; all three rows are scored by
        the same metrics code over the same bars.
      </p>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Total return</th>
              <th>Annualized</th>
              <th>Sharpe</th>
              <th>Sortino</th>
              <th>Max DD</th>
              <th>Calmar</th>
              <th>Round trips</th>
              <th>Win rate</th>
              <th>Profit factor</th>
              <th>Final equity</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((key) => {
              const row = metrics[key];
              return (
                <tr key={key}>
                  <td>{ROW_LABELS[key]}</td>
                  <td>{signed(row.total_return, pct(row.total_return))}</td>
                  <td>{signed(row.annualized_return, pct(row.annualized_return))}</td>
                  <td>{signed(row.sharpe_ratio, ratio(row.sharpe_ratio))}</td>
                  <td>{signed(row.sortino_ratio, ratio(row.sortino_ratio))}</td>
                  <td className="negative">{pct(row.max_drawdown)}</td>
                  <td>{signed(row.calmar_ratio, ratio(row.calmar_ratio))}</td>
                  <td>{count(row.total_trades)}</td>
                  <td>{pct(row.win_rate, 1)}</td>
                  <td>{ratio(row.profit_factor)}</td>
                  <td>{money(row.final_value)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

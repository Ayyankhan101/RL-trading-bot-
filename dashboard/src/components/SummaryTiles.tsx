import type { StrategyMetrics } from '../types';
import { count, money, pct, ratio } from '../lib/format';

interface Props {
  agent: StrategyMetrics;
}

/** Headline figures for the agent, all sourced from metrics.json. */
export default function SummaryTiles({ agent }: Props) {
  const tiles = [
    { label: 'Total return', value: pct(agent.total_return), signed: agent.total_return },
    { label: 'Annualized', value: pct(agent.annualized_return), signed: agent.annualized_return },
    { label: 'Sharpe', value: ratio(agent.sharpe_ratio), signed: agent.sharpe_ratio },
    { label: 'Max drawdown', value: pct(agent.max_drawdown), signed: agent.max_drawdown },
    { label: 'Win rate', value: pct(agent.win_rate, 1), signed: null },
    { label: 'Round trips', value: count(agent.total_trades), signed: null },
    { label: 'Profit factor', value: ratio(agent.profit_factor), signed: null },
    { label: 'Final equity', value: money(agent.final_value), signed: null },
  ];

  return (
    <div className="tiles">
      {tiles.map((tile) => (
        <div className="tile" key={tile.label}>
          <div className="label">{tile.label}</div>
          <div
            className={
              tile.signed === null ? 'value' : `value ${tile.signed >= 0 ? 'positive' : 'negative'}`
            }
          >
            {tile.value}
          </div>
        </div>
      ))}
    </div>
  );
}

import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { LiveBundle } from '../types';
import { count, money, pct, ratio, shortDate } from '../lib/format';

interface Props {
  live: LiveBundle | null;
}

/**
 * Live paper-trading panel.
 *
 * Renders results/live/ exactly as written. When the bot has never run, it says
 * so rather than showing an empty chart that reads like a flat account.
 */
export default function LivePanel({ live }: Props) {
  if (!live) {
    return (
      <div className="panel">
        <h2>Live paper trading</h2>
        <p className="hint">The live bot has not run yet.</p>
        <pre className="inline-cmd">python live_trade.py --once --catch-up 400</pre>
      </div>
    );
  }

  const { status, equity, promotions } = live;
  const learning = status.online_learning;
  const perBar = learning.per_bar;
  const benchmark = status.buy_hold_return_since_start;
  const beatsBenchmark = benchmark !== null && status.total_return > benchmark;

  return (
    <div className="panel">
      <h2>Live paper trading</h2>
      <p className="hint">
        {status.symbol} {status.interval} bars from {status.source} · {status.mode} account ·{' '}
        {status.model_origin} model · last bar {status.last_bar} at {money(status.last_price)} ·
        updated {status.updated_at}
      </p>

      <div className="tiles" style={{ marginTop: 0 }}>
        <div className="tile">
          <div className="label">Equity</div>
          <div className="value">{money(status.equity)}</div>
        </div>
        <div className="tile">
          <div className="label">Return</div>
          <div className={`value ${status.total_return >= 0 ? 'positive' : 'negative'}`}>
            {pct(status.total_return)}
          </div>
        </div>
        <div className="tile">
          <div className="label">Buy &amp; hold, same window</div>
          <div className={`value ${(benchmark ?? 0) >= 0 ? 'positive' : 'negative'}`}>
            {pct(benchmark)}
          </div>
        </div>
        <div className="tile">
          <div className="label">vs benchmark</div>
          <div className={`value ${beatsBenchmark ? 'positive' : 'negative'}`}>
            {benchmark === null ? '—' : pct(status.total_return - benchmark)}
          </div>
        </div>
        <div className="tile">
          <div className="label">Position</div>
          <div className="value">
            {status.btc_held > 0 ? `${ratio(status.btc_held, 4)} BTC` : 'flat'}
          </div>
        </div>
        <div className="tile">
          <div className="label">Round trips</div>
          <div className="value">{count(status.round_trips)}</div>
        </div>
        <div className="tile">
          <div className="label">Win rate</div>
          <div className="value">{pct(status.win_rate, 1)}</div>
        </div>
        <div className="tile">
          <div className="label">Drawdown</div>
          <div className="value negative">{pct(status.drawdown)}</div>
        </div>
      </div>

      {equity.length > 1 && (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={equity} margin={{ top: 20, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="#232a3a" strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              type="number"
              domain={['dataMin', 'dataMax']}
              scale="time"
              tickFormatter={shortDate}
              stroke="#8d97ad"
              fontSize={11}
            />
            <YAxis
              stroke="#8d97ad"
              fontSize={11}
              tickFormatter={(value: number) => money(value)}
              width={80}
            />
            <Tooltip
              contentStyle={{ background: '#131823', border: '1px solid #232a3a', borderRadius: 8 }}
              labelFormatter={(value) => shortDate(value as number)}
              formatter={(value: number, name: string) => [money(value), name]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="equity" name="Paper account" stroke="#4cc9f0"
                  dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="buy_hold" name="Buy & hold" stroke="#f4a261"
                  dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}

      <h2 style={{ marginTop: 24 }}>Online learning</h2>
      <p className="hint">
        {perBar?.enabled ? (
          <>
            <strong>{count(perBar.gradient_updates)}</strong> gradient updates over{' '}
            <strong>{count(perBar.bars_observed)}</strong> bars — one per closed bar, applied
            to a shadow model. Next scoring in {perBar.bars_until_gate} of{' '}
            {perBar.gate_every_bars} bars.{' '}
          </>
        ) : (
          <>Per-bar learning disabled. </>
        )}
        {learning.cycles} gate {learning.cycles === 1 ? 'decision' : 'decisions'},{' '}
        {learning.promotions} promoted. A candidate only replaces the live model if it beats
        it on scored bars — learning constantly is not the same as improving constantly.
      </p>

      {promotions.length > 0 && (
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Source</th>
                <th>Metric</th>
                <th>Champion</th>
                <th>Challenger</th>
                <th>Margin needed</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {promotions.slice(-12).reverse().map((record, index) => (
                <tr key={`${record.timestamp}-${index}`}>
                  <td>{record.timestamp}</td>
                  <td>{record.source ?? 'retrain'}</td>
                  <td>{record.metric}</td>
                  <td>{ratio(record.champion_score)}</td>
                  <td>{ratio(record.challenger_score)}</td>
                  <td>{ratio(record.min_improvement)}</td>
                  <td className={record.promoted ? 'positive' : ''}>
                    {record.promoted ? 'promoted' : 'kept champion'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

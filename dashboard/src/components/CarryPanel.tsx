import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { CarryBundle } from '../types';
import { count, money, pct, ratio, shortDate } from '../lib/format';

interface Props {
  carry: CarryBundle | null;
}

/**
 * Funding-carry paper account.
 *
 * This is the only strategy here with a positive walk-forward result, because
 * it does not forecast price: it holds long spot against short perpetual and
 * collects the funding longs pay shorts every 8 hours.
 */
export default function CarryPanel({ carry }: Props) {
  if (!carry) {
    return (
      <div className="panel">
        <h2>Funding carry (delta-neutral)</h2>
        <p className="hint">The carry bot has not run yet.</p>
        <pre className="inline-cmd">python live_basis.py --once</pre>
      </div>
    );
  }

  const { status, equity } = carry;
  const weekly = status.weekly;

  return (
    <div className="panel">
      <h2>Funding carry (delta-neutral)</h2>
      <p className="hint">
        Long spot / short perpetual on {status.symbol} at {ratio(status.leverage, 0)}x ·{' '}
        {status.mode} account · {status.in_position ? 'in position' : 'flat'} ·
        held {count(status.intervals_in_position)} of {count(status.intervals_seen)} funding
        intervals · updated {status.updated_at}
      </p>

      <div className="tiles" style={{ marginTop: 0 }}>
        <div className="tile">
          <div className="label">Equity</div>
          <div className="value">{money(status.equity)}</div>
        </div>
        <div className="tile">
          <div className="label">Total return</div>
          <div className={`value ${status.total_return >= 0 ? 'positive' : 'negative'}`}>
            {pct(status.total_return, 3)}
          </div>
        </div>
        <div className="tile">
          <div className="label">Positive weeks</div>
          <div className="value positive">
            {weekly ? pct(weekly.positive_week_rate, 1) : '—'}
          </div>
        </div>
        <div className="tile">
          <div className="label">Mean week</div>
          <div className="value">
            {weekly ? pct(weekly.mean_weekly_return, 3) : '—'}
          </div>
        </div>
        <div className="tile">
          <div className="label">Funding collected</div>
          <div className="value positive">{money(status.funding_collected)}</div>
        </div>
        <div className="tile">
          <div className="label">Basis P&amp;L</div>
          <div className="value">{money(status.basis_pnl)}</div>
        </div>
        <div className="tile">
          <div className="label">Fees paid</div>
          <div className="value negative">{money(status.costs_paid)}</div>
        </div>
        <div className="tile">
          <div className="label">Liquidations</div>
          <div className={`value ${status.liquidations ? 'negative' : 'positive'}`}>
            {count(status.liquidations)}
          </div>
        </div>
      </div>

      {equity.length > 1 && (
        <ResponsiveContainer width="100%" height={240}>
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
            <YAxis stroke="#8d97ad" fontSize={11} width={80}
                   tickFormatter={(value: number) => money(value)} />
            <Tooltip
              contentStyle={{ background: '#131823', border: '1px solid #232a3a', borderRadius: 8 }}
              labelFormatter={(value) => shortDate(value as number)}
              formatter={(value: number) => [money(value), 'Equity']}
            />
            <Line type="monotone" dataKey="equity" stroke="#34d399" dot={false}
                  strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}

      <p className="hint" style={{ marginTop: 16 }}>
        Current funding {pct(status.last_funding_rate, 5)} per 8h
        ({pct(status.annualized_funding, 2)} annualized) · basis{' '}
        {pct(status.current_basis, 4)} · liquidated by a{' '}
        {pct(status.liquidation_move, 1)} adverse move. No price forecast is involved;
        price moves cancel between the two legs.
      </p>
    </div>
  );
}

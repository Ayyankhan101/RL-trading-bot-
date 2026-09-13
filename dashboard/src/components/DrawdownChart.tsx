import { useMemo } from 'react';
import {
  Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { EquityPoint } from '../types';
import { drawdownSeries, pct, shortDate } from '../lib/format';

interface Props {
  equity: EquityPoint[];
}

/** Underwater plot: how far below its own peak each strategy was at every bar. */
export default function DrawdownChart({ equity }: Props) {
  const data = useMemo(() => {
    const agent = drawdownSeries(equity.map((point) => point.rl_agent));
    const hold = drawdownSeries(equity.map((point) => point.buy_and_hold ?? Number.NaN));

    return equity.map((point, index) => ({
      time: point.time,
      agent: agent[index],
      hold: Number.isFinite(hold[index]) ? hold[index] : null,
    }));
  }, [equity]);

  return (
    <div className="panel">
      <h2>Drawdown</h2>
      <p className="hint">Peak-to-trough decline, recomputed from the equity curve above.</p>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
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
            tickFormatter={(value: number) => pct(value, 0)}
            width={60}
          />
          <Tooltip
            contentStyle={{ background: '#131823', border: '1px solid #232a3a', borderRadius: 8 }}
            labelFormatter={(value) => shortDate(value as number)}
            formatter={(value: number, name: string) => [pct(value), name]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area type="monotone" dataKey="agent" name="RL agent" stroke="#4cc9f0"
                fill="#4cc9f0" fillOpacity={0.18} isAnimationActive={false} />
          <Area type="monotone" dataKey="hold" name="Buy & hold" stroke="#f4a261"
                fill="#f4a261" fillOpacity={0.12} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

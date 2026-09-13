import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { EquityPoint } from '../types';
import { money, shortDate } from '../lib/format';

interface Props {
  equity: EquityPoint[];
}

/** Out-of-sample equity of the agent against both baselines, same axes. */
export default function EquityChart({ equity }: Props) {
  return (
    <div className="panel">
      <h2>Equity curve (out-of-sample)</h2>
      <p className="hint">
        Every series starts at the same balance and pays the same fees and slippage.
      </p>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={equity} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
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
          <Line type="monotone" dataKey="rl_agent" name="RL agent" stroke="#4cc9f0"
                dot={false} strokeWidth={2} isAnimationActive={false} />
          <Line type="monotone" dataKey="buy_and_hold" name="Buy & hold" stroke="#f4a261"
                dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="rsi_strategy" name="RSI strategy" stroke="#a78bfa"
                dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="mean_reversion" name="Mean reversion (rule)"
                stroke="#34d399" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

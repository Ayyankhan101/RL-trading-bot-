import { useMemo } from 'react';
import {
  Bar, BarChart, Cell, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { RoundTrip } from '../types';
import { pct } from '../lib/format';

interface Props {
  trades: RoundTrip[];
}

const BUCKET_COUNT = 21;

/** Histogram of round-trip returns. Each bar is completed trades, not orders. */
export default function TradeDistribution({ trades }: Props) {
  const buckets = useMemo(() => {
    if (trades.length === 0) return [];

    const returns = trades.map((trade) => trade.return_pct);
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const span = max - min || 0.01;
    const width = span / BUCKET_COUNT;

    const counts = Array.from({ length: BUCKET_COUNT }, (_, index) => ({
      center: min + width * (index + 0.5),
      count: 0,
    }));

    returns.forEach((value) => {
      const index = Math.min(BUCKET_COUNT - 1, Math.floor((value - min) / width));
      counts[index].count += 1;
    });

    return counts;
  }, [trades]);

  if (buckets.length === 0) {
    return (
      <div className="panel">
        <h2>Round-trip returns</h2>
        <p className="hint">The agent completed no round trips in this run.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Round-trip returns</h2>
      <p className="hint">
        {trades.length} completed entry/exit pairs. Green bars closed above the entry price.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={buckets} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="#232a3a" strokeDasharray="3 3" />
          <XAxis
            dataKey="center"
            stroke="#8d97ad"
            fontSize={11}
            tickFormatter={(value: number) => pct(value, 1)}
          />
          <YAxis stroke="#8d97ad" fontSize={11} width={40} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#131823', border: '1px solid #232a3a', borderRadius: 8 }}
            labelFormatter={(value) => `Return ≈ ${pct(value as number, 1)}`}
            formatter={(value: number) => [`${value} trades`, 'Count']}
          />
          <Bar dataKey="count" isAnimationActive={false}>
            {buckets.map((bucket) => (
              <Cell key={bucket.center} fill={bucket.center >= 0 ? '#34d399' : '#f87171'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

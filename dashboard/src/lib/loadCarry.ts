import { num, parseCsv } from './csv';
import type { CarryBundle, CarryPoint, CarryStatus } from '../types';

const BASE = 'results/live_basis';

async function fetchOptional(name: string): Promise<string | null> {
  const response = await fetch(`${BASE}/${name}`);
  return response.ok ? response.text() : null;
}

/** Returns null when the carry bot has never run — the page then says so. */
export async function loadCarry(): Promise<CarryBundle | null> {
  const statusText = await fetchOptional('status.json');
  if (!statusText) return null;

  const equityCsv = await fetchOptional('equity.csv');
  const equity: CarryPoint[] = equityCsv
    ? parseCsv(equityCsv)
        .map((row) => {
          const value = num(row.equity);
          if (value === null) return null;
          return {
            timestamp: row.timestamp,
            time: new Date(row.timestamp).getTime(),
            equity: value,
            funding_rate: num(row.funding_rate) ?? 0,
            basis: num(row.basis) ?? 0,
            in_position: row.in_position === 'True' || row.in_position === 'true',
          } satisfies CarryPoint;
        })
        .filter((point): point is CarryPoint => point !== null)
    : [];

  return { status: JSON.parse(statusText) as CarryStatus, equity };
}

import { num, parseCsv } from './csv';
import type { LiveBundle, LiveEquityPoint, LiveStatus, PromotionRecord } from '../types';

const BASE = 'results/live';

async function fetchOptional(name: string): Promise<string | null> {
  const response = await fetch(`${BASE}/${name}`);
  return response.ok ? response.text() : null;
}

/**
 * Rebase the bar prices into a buy & hold curve starting at the same balance,
 * so the live account is judged against the alternative of doing nothing.
 */
function toLiveEquity(rows: Record<string, string>[]): LiveEquityPoint[] {
  const points: LiveEquityPoint[] = [];
  let firstPrice: number | null = null;
  let startEquity: number | null = null;

  rows.forEach((row) => {
    const equity = num(row.equity);
    const price = num(row.price);
    if (equity === null || price === null) return;

    if (firstPrice === null) {
      firstPrice = price;
      startEquity = equity;
    }

    points.push({
      timestamp: row.timestamp,
      time: new Date(row.timestamp).getTime(),
      equity,
      buy_hold: (startEquity ?? equity) * (price / (firstPrice ?? price)),
      price,
    });
  });

  return points;
}

/** Returns null when the bot has never run — the page then says so. */
export async function loadLive(): Promise<LiveBundle | null> {
  const statusText = await fetchOptional('status.json');
  if (!statusText) return null;

  const [equityCsv, promotionsJson] = await Promise.all([
    fetchOptional('equity.csv'),
    fetchOptional('promotions.json'),
  ]);

  return {
    status: JSON.parse(statusText) as LiveStatus,
    equity: equityCsv ? toLiveEquity(parseCsv(equityCsv)) : [],
    promotions: promotionsJson ? (JSON.parse(promotionsJson) as PromotionRecord[]) : [],
  };
}

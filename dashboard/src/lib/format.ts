const NA = '—';

export function pct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NA;
  return `${(value * 100).toFixed(digits)}%`;
}

export function ratio(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NA;
  return value.toFixed(digits);
}

export function money(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NA;
  return value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  });
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NA;
  return value.toLocaleString('en-US');
}

export function shortDate(value: string | number): string {
  const date = typeof value === 'number' ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' });
}

/** Running peak-to-trough drawdown of an equity series. */
export function drawdownSeries(values: number[]): number[] {
  let peak = Number.NEGATIVE_INFINITY;
  return values.map((value) => {
    peak = Math.max(peak, value);
    return peak > 0 ? (value - peak) / peak : 0;
  });
}

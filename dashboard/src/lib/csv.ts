/**
 * Minimal CSV reader for the artifacts this dashboard consumes.
 *
 * backtest.py writes plain numeric columns and ISO timestamps - no embedded
 * commas, quotes or newlines - so a full CSV parser would be dead weight.
 */

export function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const header = lines[0].split(',');

  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    const row: Record<string, string> = {};
    header.forEach((key, index) => {
      row[key] = cells[index] ?? '';
    });
    return row;
  });
}

/** Parse a numeric cell, treating blanks and non-numbers as null. */
export function num(value: string | undefined): number | null {
  if (value === undefined || value.trim() === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// Copy the backtest artifacts into public/ so Vite can serve them.
// The dashboard has no backend and invents no data: if results/ is empty, the
// page says so rather than rendering placeholders.
import { cp, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', '..', 'results');
const destination = join(here, '..', 'public', 'results');

if (!existsSync(source)) {
  console.warn(`[sync] No results/ directory at ${source}.`);
  console.warn('[sync] Run:  python backtest.py --data data/BTC.csv');
  await mkdir(destination, { recursive: true });
  process.exit(0);
}

await mkdir(destination, { recursive: true });
await cp(source, destination, { recursive: true });

const files = await readdir(destination);
console.log(`[sync] Copied ${files.length} artifact(s): ${files.join(', ')}`);

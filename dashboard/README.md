# Dashboard

Static Vite + React + TypeScript dashboard for the backtest results.

Every figure on the page is read from `../results/`, which `backtest.py` writes.
There is no backend, no sample data and no hardcoded metric: if the artifacts are
missing, the page tells you to run the backtest.

```bash
python backtest.py --data data/BTC.csv   # writes ../results/
npm install
npm run dev                              # http://localhost:5173
npm run build                            # static bundle in dist/
```

`npm run sync` (invoked by `dev` and `build`) copies `../results/` into
`public/results/`.

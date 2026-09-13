/**
 * Shapes of the artifacts written by backtest.py.
 *
 * These interfaces are the contract between the Python pipeline and this page.
 * If the artifact schema changes, the build breaks here rather than the page
 * silently rendering a wrong number.
 */

export interface StrategyMetrics {
  label: string;
  initial_value: number;
  final_value: number;
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  var_95: number;
  cvar_95: number;
  n_periods: number;
  periods_per_year: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number | null;
  expectancy?: number;
  winning_trades?: number;
  losing_trades?: number;
  avg_winning_trade?: number;
  avg_losing_trade?: number;
  largest_win?: number;
  largest_loss?: number;
}

export type StrategyKey = 'rl_agent' | 'buy_and_hold' | 'rsi_strategy' | 'mean_reversion';

export type MetricsFile = Record<StrategyKey, StrategyMetrics>;

export interface FoldReport {
  fold: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  win_rate: number;
}

export interface RunMeta {
  generated_at: string;
  git_sha: string | null;
  seed: number | null;
  variant: string;
  data_rows: number;
  data_start: string;
  data_end: string;
  oos_start: string;
  oos_end: string;
  periods_per_year: number;
  transaction_cost: number;
  slippage: number;
  folds: FoldReport[];
}

export interface EquityPoint {
  timestamp: string;
  time: number;
  rl_agent: number;
  buy_and_hold: number | null;
  rsi_strategy: number | null;
  mean_reversion: number | null;
}

export interface RoundTrip {
  entry_timestamp: string;
  exit_timestamp: string;
  entry_price: number;
  exit_price: number;
  amount: number;
  pnl: number;
  return_pct: number;
  bars_held: number;
  exit_reason: string;
}

export interface ResultsBundle {
  metrics: MetricsFile;
  meta: RunMeta;
  equity: EquityPoint[];
  trades: RoundTrip[];
}

/** Live paper-trading artifacts, written by live_trade.py into results/live/. */
export interface PromotionRecord {
  timestamp: string;
  /** Which learner produced this candidate: per-bar shadow, or weekly retrain. */
  source?: 'shadow' | 'retrain';
  last_bar: string;
  metric: string;
  champion_score: number;
  challenger_score: number;
  min_improvement: number;
  promoted: boolean;
  bars_observed?: number;
  gradient_updates?: number;
  train_window?: [string, string];
  eval_window: [string, string];
  champion_return: number | null;
  challenger_return: number | null;
}

export interface LiveStatus {
  updated_at: string;
  started_at: string | null;
  symbol: string;
  interval: string;
  source: string;
  mode: string;
  model_origin: string;
  last_bar: string;
  last_price: number;
  new_bars: number;
  buy_hold_return_since_start: number | null;
  equity: number;
  balance: number;
  btc_held: number;
  position_value: number;
  entry_price: number | null;
  unrealized_pct: number;
  total_return: number;
  drawdown: number;
  round_trips: number;
  win_rate: number;
  fills: number;
  halted: boolean;
  online_learning: {
    enabled: boolean;
    bars_since_retrain: number;
    retrain_every_bars: number;
    promotions: number;
    cycles: number;
    last_decision: PromotionRecord | null;
    per_bar?: {
      enabled: boolean;
      bars_observed: number;
      gradient_updates: number;
      bars_until_gate: number;
      gate_every_bars: number;
    };
  };
}

export interface LiveEquityPoint {
  timestamp: string;
  time: number;
  equity: number;
  buy_hold: number;
  price: number;
}

export interface LiveBundle {
  status: LiveStatus;
  equity: LiveEquityPoint[];
  promotions: PromotionRecord[];
}

/** Funding-carry paper account, written by live_basis.py. */
export interface CarryStatus {
  updated_at: string;
  strategy: string;
  mode: string;
  symbol: string;
  leverage: number;
  equity: number;
  total_return: number;
  in_position: boolean;
  entry_timestamp: string | null;
  funding_collected: number;
  basis_pnl: number;
  costs_paid: number;
  intervals_in_position: number;
  intervals_seen: number;
  liquidations: number;
  last_interval: string | null;
  last_funding_rate: number;
  trailing_funding: number;
  current_basis: number;
  annualized_funding: number;
  liquidation_move: number;
  weekly: {
    weeks: number;
    positive_weeks: number;
    positive_week_rate: number;
    mean_weekly_return: number;
    median_weekly_return: number;
    worst_week: number;
    best_week: number;
  } | null;
}

export interface CarryPoint {
  timestamp: string;
  time: number;
  equity: number;
  funding_rate: number;
  basis: number;
  in_position: boolean;
}

export interface CarryBundle {
  status: CarryStatus;
  equity: CarryPoint[];
}

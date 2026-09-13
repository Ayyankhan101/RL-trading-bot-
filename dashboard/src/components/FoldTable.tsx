import type { FoldReport } from '../types';
import { count, pct, ratio, shortDate } from '../lib/format';

interface Props {
  folds: FoldReport[];
}

/**
 * Per-fold out-of-sample results.
 *
 * The headline number is a single path; the spread across folds is what says
 * whether it was skill or one lucky window.
 */
export default function FoldTable({ folds }: Props) {
  if (folds.length === 0) return null;

  return (
    <div className="panel">
      <h2>Walk-forward folds</h2>
      <p className="hint">
        Each fold trains only on bars preceding its test window, with a purge gap between them.
      </p>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Fold</th>
              <th>Train through</th>
              <th>Test window</th>
              <th>Return</th>
              <th>Sharpe</th>
              <th>Max DD</th>
              <th>Round trips</th>
              <th>Win rate</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((fold) => (
              <tr key={fold.fold}>
                <td>{fold.fold}</td>
                <td>{shortDate(fold.train_end)}</td>
                <td>{shortDate(fold.test_start)} → {shortDate(fold.test_end)}</td>
                <td className={fold.total_return >= 0 ? 'positive' : 'negative'}>
                  {pct(fold.total_return)}
                </td>
                <td>{ratio(fold.sharpe_ratio)}</td>
                <td className="negative">{pct(fold.max_drawdown)}</td>
                <td>{count(fold.total_trades)}</td>
                <td>{pct(fold.win_rate, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

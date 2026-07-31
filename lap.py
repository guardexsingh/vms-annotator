"""Minimal lapx-compatible assignment API backed by locked SciPy."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "0.1-scipy-compat"


def lapjv(cost_matrix, extend_cost: bool = False, cost_limit: float = np.inf,
          return_cost: bool = True):
    """Return lapx-style thresholded row/column assignments."""
    del extend_cost  # SciPy supports rectangular matrices directly.
    costs = np.asarray(cost_matrix, dtype=float)
    rows, columns = costs.shape
    row_assignment = np.full(rows, -1, dtype=int)
    column_assignment = np.full(columns, -1, dtype=int)
    total = 0.0
    if rows and columns:
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        for row, column in zip(assigned_rows, assigned_columns):
            cost = float(costs[row, column])
            if cost <= cost_limit:
                row_assignment[row] = column
                column_assignment[column] = row
                total += cost
    return (total if return_cost else 0.0), row_assignment, column_assignment

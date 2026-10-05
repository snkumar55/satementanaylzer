from __future__ import annotations

import pandas as pd

def merchant_summary(
    spending: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return ranked spend, count, and average transaction per merchant."""
    if spending.empty:
        return pd.DataFrame(columns=["Transactions", "Total", "Average"])
    summary = (
        spending.groupby("Merchant", dropna=False)
        .agg(
            Transactions=("Withdrawal", "size"),
            Total=("Withdrawal", "sum"),
            Average=("Withdrawal", "mean"),
        )
        .sort_values(["Total", "Transactions"], ascending=False)
    )
    return summary.head(limit) if limit is not None else summary

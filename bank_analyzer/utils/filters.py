from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

EXCLUSION_WARNING_RATIO = 0.25

def apply_transaction_exclusions(
    df: pd.DataFrame,
    exclusion_keywords: str | Iterable[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transaction rows into matching exclusions and rows to analyze."""
    if isinstance(exclusion_keywords, str):
        keywords = re.split(r"[,\n]", exclusion_keywords)
    else:
        keywords = []
        for keyword in exclusion_keywords or []:
            keywords.extend(re.split(r"[,\n]", str(keyword)))
    keywords = list(dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip()))

    if df.empty or not keywords:
        return df.copy(), df.iloc[0:0].copy()

    description_columns = [
        column for column in df.columns
        if column.casefold() in {"description", "narration"}
    ]
    if not description_columns:
        raise ValueError("Transactions must include a Description or Narration column.")

    descriptions = (
        df[description_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    excluded_mask = descriptions.str.contains(pattern, case=False, regex=True, na=False)
    return df.loc[~excluded_mask].copy(), df.loc[excluded_mask].copy()

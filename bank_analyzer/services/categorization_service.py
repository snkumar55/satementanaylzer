from __future__ import annotations

import pandas as pd


def categorize_description(description: str, category_rules: dict[str, list[str]]) -> str:
    """Assign a description using the analyzer's existing ordered keyword rules."""
    if not isinstance(description, str):
        return "Other"
    normalized = description.casefold()
    for category, keywords in category_rules.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    return "Other"


def normalize_categories(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize existing analyzer categories without replacing its rules."""
    normalized = records.copy()
    if "Category" not in normalized:
        normalized["Category"] = "Other"
    normalized["Category"] = (
        normalized["Category"].fillna("").astype(str).str.strip().replace("", "Other")
    )
    return normalized

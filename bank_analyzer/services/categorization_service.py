from __future__ import annotations

import pandas as pd

CATEGORY_RULES = {
    "Food": [
        "restaurant", "restaurants", "cafe", "coffee", "diner", "canteen", "food", "kitchen",
        "truffles", "bamboo", "curry", "tiffin", "madras", "thatha", "udupi", "zomato",
        "swiggy", "eatfit", "dominos", "pizza", "burger", "kfc", "mcdonald", "starbucks",
        "biryani", "bakery", "sweet", "juice", "chaat", "mess", "bhavan", "a2b",
        "sangeetha", "empire", "meghana", "food court",
    ],
    "Groceries": [
        "grocery", "groceries", "supermarket", "hyper market", "all day", "zepto",
        "blinkit", "amazonpaygrocery",
    ],
    "Bills": [
        "bill", "billpay", "utility", "fastag", "airtel", "jio", "vi ", "bsnl",
        "bescom", "bwssb", "netflix", "recharge", "airtelpayments",
    ],
    "Transport": [
        "railways", "metro", "bus", "uber", "ola", "rapido", "taxi", "bm tc",
        "bmtc", "petrol", "fuel",
    ],
    "Entertainment": [
        "movie", "spotify", "bookmyshow", "primevideo", "hotstar", "sonyliv",
        "pink berry",
    ],
    "EMI": ["emi", "chq", "cheque"],
    "Refunds": ["refund", "refunds", "razorpay", "amazon.refunds"],
    "Transfers": ["tpt", "transfer", "imps", "neft", "p2p", "self", "transfer to"],
    "Other": [],
}


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

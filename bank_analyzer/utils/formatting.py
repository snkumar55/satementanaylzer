from __future__ import annotations

CURRENCY_SYMBOL = "₹"

def format_currency(value: float) -> str:
    """Format a numeric amount using the app's INR display convention."""
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(amount):,.2f}"

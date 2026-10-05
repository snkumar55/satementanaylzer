from __future__ import annotations

import pandas as pd

from bank_analyzer.services.categorization_service import normalize_categories
from bank_analyzer.services.transfer_service import annotate_transfer_types
from bank_analyzer.utils.formatting import format_currency

CHART_COLORS = {
    "credit": "#16856d",
    "spending": "#d45a58",
    "transfer": "#5377c5",
    "insight": "#d88a28",
    "neutral": "#64748b",
}

def spending_intelligence_data(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized transactions and external debit rows for analysis."""
    records = pd.DataFrame(engine.raw_transactions)
    if records.empty:
        return records, records

    records["Date"] = pd.to_datetime(records["Date"], errors="coerce")
    for column in ("Withdrawal", "Deposit", "Amount"):
        records[column] = pd.to_numeric(records[column], errors="coerce").fillna(0.0)
    records = records.dropna(subset=["Date"])
    records = annotate_transfer_types(
        records, getattr(engine, "account_holder_names", ())
    )
    records = normalize_categories(records)

    spending = records[(records["Withdrawal"] > 0) & ~records["SelfTransfer"]].copy()
    spending["MonthPeriod"] = spending["Date"].dt.to_period("M")
    spending["Month"] = spending["MonthPeriod"].dt.to_timestamp()
    spending["Merchant"] = spending["Merchant"].replace("", "Unknown").fillna("Unknown")
    spending["Category"] = spending["Category"].replace("", "Other").fillna("Other")
    return records, spending

def dashboard_summary(
    records: pd.DataFrame,
    account_holder_names: tuple[str, ...] = (),
) -> dict[str, object]:
    """Calculate overview KPIs once for consistent presentation."""
    if records.empty:
        return {
            "transaction_count": 0,
            "money_in": 0.0,
            "money_out": 0.0,
            "external_spending": 0.0,
            "refunds": 0.0,
            "transfers": 0.0,
            "net_spending": 0.0,
            "period_start": None,
            "period_end": None,
            "records": records.copy(),
        }
    annotated = annotate_transfer_types(records, account_holder_names)
    dates = pd.to_datetime(annotated["Date"], errors="coerce").dropna()
    external_spending = float(annotated.loc[~annotated["SelfTransfer"], "Withdrawal"].sum())
    refunds = float(annotated["ReturnedAmount"].sum())
    return {
        "transaction_count": len(annotated),
        "money_in": float(annotated["Deposit"].sum()),
        "money_out": float(annotated["Withdrawal"].sum()),
        "external_spending": external_spending,
        "refunds": refunds,
        "transfers": float(annotated["SelfTransferAmount"].sum()),
        "net_spending": external_spending - refunds,
        "period_start": dates.min() if not dates.empty else None,
        "period_end": dates.max() if not dates.empty else None,
        "records": annotated,
    }

def transaction_totals(
    records: pd.DataFrame,
    account_holder_names: tuple[str, ...] = (),
) -> dict[str, float | int]:
    """Calculate filtered transaction KPIs and transfer/refund reconciliation."""
    annotated = annotate_transfer_types(records, account_holder_names)
    deposits = float(annotated["Deposit"].sum())
    withdrawals = float(annotated["Withdrawal"].sum())
    refunds = float(annotated["ReturnedAmount"].sum())
    transfers = float(annotated["SelfTransferAmount"].sum())
    external_withdrawals = float(
        annotated.loc[~annotated["SelfTransfer"], "Withdrawal"].sum()
    )
    return {
        "transaction_count": len(annotated),
        "money_in": deposits,
        "money_out": withdrawals,
        "refunds": refunds,
        "transfers": transfers,
        "sent_to_others": float(
            annotated.loc[annotated["SentToOthers"], "Withdrawal"].sum()
        ),
        "external_withdrawals": external_withdrawals,
        "net_spending": external_withdrawals - refunds,
    }

def spending_health_score(
    spending: pd.DataFrame,
    category_totals: pd.DataFrame,
    merchant_totals: pd.DataFrame,
    monthly: pd.DataFrame,
) -> tuple[int, list[str], dict[str, float]]:
    if spending.empty:
        return 0, [], {}

    total = float(spending["Withdrawal"].sum())
    top_five_share = float(category_totals.head(5)["Total"].sum() / total) if total else 0.0
    concentration_score = 25 * (1 - min(top_five_share / 0.85, 1))

    median_amount = float(spending["Withdrawal"].median())
    large_limit = max(median_amount * 2, float(spending["Withdrawal"].quantile(0.9)))
    large_count = int((spending["Withdrawal"] >= large_limit).sum()) if large_limit > 0 else 0
    large_share = large_count / max(1, len(spending))
    large_purchase_score = 25 * (1 - min(large_share / 0.25, 1))

    recurring_merchants = spending.groupby("Merchant").agg(
        Count=("Withdrawal", "size"),
        Months=("MonthPeriod", "nunique"),
        Total=("Withdrawal", "sum"),
    )
    recurring_merchants = recurring_merchants[
        (recurring_merchants["Months"] >= 2) | (recurring_merchants["Count"] >= 3)
    ]
    recurring_share = float(recurring_merchants["Total"].sum() / total) if total else 0.0
    recurring_score = 25 * (1 - min(recurring_share / 0.5, 1))

    if len(monthly) >= 2 and float(monthly["Total"].mean()) > 0:
        coefficient_variation = float(monthly["Total"].std(ddof=0) / monthly["Total"].mean())
        stability_score = 25 * (1 - min(coefficient_variation, 1))
        stability_driver = (
            f"Monthly spending varies by {coefficient_variation:.0%} of its average "
            f"({stability_score:.0f}/25 stability points)."
        )
    else:
        stability_score = 12.5
        stability_driver = "Only one spending month is available; stability receives a neutral score."

    score = round(concentration_score + large_purchase_score + recurring_score + stability_score)
    drivers = [
        f"Top five categories are {top_five_share:.0%} of spending "
        f"({concentration_score:.0f}/25 concentration points).",
        f"{large_count} transactions are at least {format_currency(large_limit)} "
        f"({large_purchase_score:.0f}/25 large-purchase points).",
        f"Recurring-pattern merchants represent {recurring_share:.0%} of spending "
        f"({recurring_score:.0f}/25 recurring-pattern points).",
        stability_driver,
    ]
    components = {
        "Concentration": concentration_score,
        "Large purchases": large_purchase_score,
        "Recurring patterns": recurring_score,
        "Monthly stability": stability_score,
    }
    return score, drivers, components

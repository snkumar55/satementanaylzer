from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from bank_analyzer.services.categorization_service import (
    CATEGORY_RULES,
    categorize_description,
)
from bank_analyzer.services.forecast_service import compute_forecast


class StreamlitAnalyzer:
    """Pure-Python statement processor and query adapter for the Streamlit UI."""

    def __init__(self) -> None:
        self.raw_transactions: list[dict] = []
        self.total_income = "₹0.00"
        self.total_expenses = "₹0.00"
        self.monthly_chart_data: list[dict] = []
        self.monthly_recurring_table: list[list[str]] = []
        self.forecast_table: list[list[str]] = []
        self.chat_messages = [
            ["assistant", "Ask about payments, deposits, repeated merchants, or monthly totals."]
        ]
        self.account_holder_names: tuple[str, ...] = ()
        self.statement_filename = "Uploaded statement"
        self.forecast_model = "exp_smoothing"
        self.exp_smoothing_trend = "add"
        self.exp_smoothing_smoothing_level = 0.2

    @staticmethod
    def _parse_amount(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """Parse ISO and common day-first bank-statement date formats."""
        values = series.astype(str).str.strip()
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        for date_format in (
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d %b %Y",
        ):
            missing = parsed.isna()
            parsed.loc[missing] = pd.to_datetime(
                values.loc[missing],
                format=date_format,
                errors="coerce",
            )
        missing = parsed.isna()
        parsed.loc[missing] = pd.to_datetime(
            values.loc[missing],
            errors="coerce",
            dayfirst=True,
        )
        return parsed

    @staticmethod
    def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        lookup = {str(column).strip().casefold(): column for column in frame.columns}
        return next((lookup[name.casefold()] for name in candidates if name.casefold() in lookup), None)

    @staticmethod
    def _payment_mode(description: str) -> str:
        lowered = description.casefold()
        if any(value in lowered for value in ("upi", "gpay", "google pay", "phonepe", "paytm", "bhim", "@upi", "p2m", "p2a", "vpa")):
            return "UPI"
        if any(value in lowered for value in ("pos", "debit card", "credit card", "card", "visa", "mastercard", "rupay", "swipe")):
            return "Card"
        if any(value in lowered for value in ("imps", "neft", "rtgs", "tpt", "transfer")):
            return "Transfer"
        if any(value in lowered for value in ("ach", "nach", "ecs", "autopay", "standing instruction")):
            return "Auto Debit"
        if any(value in lowered for value in ("atm", "cash withdrawal", "cash wdl")):
            return "ATM/Cash"
        return "Other"

    @staticmethod
    def _merchant_name(description: str) -> str:
        value = re.sub(r"\s+", " ", description).strip()
        value = re.sub(
            r"\b(?:upi|imps|neft|rtgs|ref(?:erence)?|txn|transaction|payment|transfer)\b",
            " ",
            value,
            flags=re.I,
        )
        value = re.sub(r"\b\d{5,}\b|[A-Z0-9._%+-]+@[A-Z0-9.-]+", " ", value, flags=re.I)
        value = re.sub(r"[^A-Za-z0-9&.' -]", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" -.")
        return value[:60].title() if value else "Unknown"

    def process_transactions(
        self,
        source: pd.DataFrame,
        clean_statement_rows: bool = True,
    ) -> None:
        """Normalize statement rows and recalculate all Streamlit analysis inputs."""
        frame = source.copy()
        frame.columns = frame.columns.astype(str).str.strip()
        if "Date" not in frame:
            raise ValueError("The statement must contain a Date column.")

        description_column = self._find_column(frame, ("Narration", "Description"))
        if description_column is None:
            description_column = frame.columns[1] if len(frame.columns) > 1 else None
        if description_column is None:
            frame["Description"] = "Unknown Transaction"
        else:
            frame["Description"] = frame[description_column].fillna("").astype(str)

        debit_column = self._find_column(
            frame, ("Withdrawal Amt.", "Withdrawal Amount", "Debit", "Dr", "Debits", "Withdrawal")
        )
        credit_column = self._find_column(
            frame, ("Deposit Amt.", "Deposit Amount", "Credit", "Cr", "Credits", "Deposit")
        )
        frame["Withdrawal"] = self._parse_amount(frame[debit_column]) if debit_column else 0.0
        frame["Deposit"] = self._parse_amount(frame[credit_column]) if credit_column else 0.0
        frame["Date"] = self._parse_dates(frame["Date"])
        frame["Amount"] = frame["Deposit"] - frame["Withdrawal"]

        if clean_statement_rows:
            description = frame["Description"].str.casefold()
            summary_rows = description.str.contains(
                r"statement summary|opening balance|closing balance|dr count|cr count",
                regex=True,
                na=False,
            )
            frame = frame.loc[~summary_rows]
        frame = frame.loc[
            frame["Date"].notna()
            & (frame["Amount"] != 0)
            & (frame[["Withdrawal", "Deposit"]].max(axis=1) < 1_000_000_000)
        ].copy()
        if frame.empty:
            self.raw_transactions = []
            self.total_income = self.total_expenses = "₹0.00"
            self.monthly_chart_data = []
            self.monthly_recurring_table = []
            self.forecast_table = []
            return

        frame["Category"] = frame["Description"].map(
            lambda description: categorize_description(description, CATEGORY_RULES)
        )
        frame["PaymentMode"] = frame["Description"].map(self._payment_mode)
        frame["Merchant"] = frame["Description"].map(self._merchant_name)
        frame["PaymentAccountDisplay"] = frame["Merchant"]
        frame["RepeatPaymentDisplay"] = frame["Merchant"]
        frame["SpendBand"] = np.select(
            [frame["Withdrawal"] < 5000, frame["Withdrawal"] < 10000],
            ["Micro < ₹5,000", "₹5,000-<₹10,000"],
            default="₹10,000 and above",
        )

        grouped = frame.assign(MonthPeriod=frame["Date"].dt.to_period("M"))
        monthly = (
            grouped.groupby("MonthPeriod")
            .agg(Deposit=("Deposit", "sum"), Withdrawal=("Withdrawal", "sum"))
            .sort_index()
        )
        monthly["Savings"] = monthly["Deposit"] - monthly["Withdrawal"]
        self.monthly_chart_data = [
            {
                "month": period.strftime("%b %Y"),
                "deposit": float(row["Deposit"]),
                "withdrawal": float(row["Withdrawal"]),
                "savings": float(row["Savings"]),
            }
            for period, row in monthly.iterrows()
        ]
        self.total_income = f"₹{frame['Deposit'].sum():,.2f}"
        self.total_expenses = f"₹{frame['Withdrawal'].sum():,.2f}"

        self.monthly_recurring_table = []
        debit_rows = grouped[grouped["Withdrawal"] > 0]
        for merchant, rows in debit_rows.groupby("Merchant"):
            if rows["MonthPeriod"].nunique() >= 2 or len(rows) >= 3:
                self.monthly_recurring_table.append(
                    [
                        merchant,
                        str(rows["Category"].mode().iloc[0]),
                        str(rows["PaymentMode"].mode().iloc[0]),
                        str(rows["MonthPeriod"].nunique()),
                        f"₹{rows['Withdrawal'].mean():,.2f}",
                        f"₹{rows['Withdrawal'].sum():,.2f}",
                        "—",
                        rows["Date"].max().strftime("%b %Y"),
                    ]
                )

        frame["Anomaly"] = "Normal"
        if len(frame) > 5:
            predictions = IsolationForest(contamination=0.07, random_state=42).fit_predict(
                frame[["Amount"]].to_numpy()
            )
            frame.loc[predictions == -1, "Anomaly"] = "Outlier"

        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
        self.raw_transactions = frame[
            [
                "Date", "Description", "Category", "PaymentMode", "SpendBand",
                "Merchant", "PaymentAccountDisplay", "RepeatPaymentDisplay",
                "Withdrawal", "Deposit", "Amount", "Anomaly",
            ]
        ].to_dict("records")
        self.forecast_table = compute_forecast(
            monthly.reset_index()[["Deposit", "Withdrawal"]],
            self.forecast_model,
            self.exp_smoothing_trend,
            self.exp_smoothing_smoothing_level,
        )

    def _chat_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.raw_transactions)
        if frame.empty:
            return frame
        frame["DateObj"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["MonthPeriod"] = frame["DateObj"].dt.to_period("M")
        frame["Year"] = frame["DateObj"].dt.year
        return frame

    @staticmethod
    def _chat_top_n(query: str, default: int = 5) -> int:
        match = re.search(r"\b(?:top|first|show)\s+(\d{1,2})\b", query, re.I)
        return max(1, min(20, int(match.group(1)))) if match else default

    def _chat_period_filter(self, frame: pd.DataFrame, query: str) -> tuple[pd.DataFrame, str]:
        latest = frame["DateObj"].max()
        year = re.search(r"\b(19\d{2}|20\d{2})\b", query)
        if year:
            result = frame[frame["DateObj"].dt.year == int(year.group(1))]
            return result, year.group(1)
        if not pd.isna(latest):
            month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", query, re.I)
            if month:
                months = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                }
                selected_month = months[month.group(1).casefold()[:3]]
                result = frame[
                    (frame["DateObj"].dt.month == selected_month)
                    & (frame["DateObj"].dt.year == latest.year)
                ]
                return result, month.group(0).title()
        return frame, "all loaded data"

    @staticmethod
    def _chat_match_records(frame: pd.DataFrame, phrase: str) -> pd.DataFrame:
        searchable_columns = [
            column for column in ("Description", "Merchant", "PaymentAccountDisplay")
            if column in frame
        ]
        text = frame[searchable_columns].fillna("").astype(str).agg(" ".join, axis=1)
        return frame.loc[text.str.contains(re.escape(phrase), case=False, na=False)]

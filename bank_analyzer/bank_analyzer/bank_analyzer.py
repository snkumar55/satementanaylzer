# bank_analyzer.py
import reflex as rx
import pandas as pd
import io
import numpy as np
import json
import os
import re
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Optional

# -------------------------
# Config / constants
# -------------------------
FILTER_STATE_FILE = "filter_state.json"
EXPORT_FOLDER = "exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Default compile-time categories (safe to iterate)
DEFAULT_BAR_CATEGORIES = ["Food", "Groceries", "Bills", "Transport", "Entertainment", "EMI", "Transfers"]
MICRO_LIMIT = 5000.0
CORE_LIMIT = 10000.0
ANALYSIS_AMOUNT_LIMIT = 10000.0
MICRO_BAND = "Micro < ₹5,000"
CORE_BAND = "₹5,000-<₹10,000"
LARGE_BAND = "₹10,000 and above"
INCOME_BAND = "Income/credit"
UNDER_LIMIT_FOCUS = "Under ₹10,000"
SPEND_FOCUS_OPTIONS = [
    UNDER_LIMIT_FOCUS,
    MICRO_BAND,
    CORE_BAND,
    "UPI micro",
    "Food & restaurants micro",
]
DETAIL_ALL_BANDS = "All below ₹10,000"
DETAIL_UNDER_5K = "Below ₹5,000"
DETAIL_5K_TO_10K = "₹5,000-₹10,000"
DETAIL_BAND_FILTER_OPTIONS = [DETAIL_ALL_BANDS, DETAIL_UNDER_5K, DETAIL_5K_TO_10K]
CATEGORY_COLORS = {
    "Food": "#f97316",
    "Groceries": "#22c55e",
    "Bills": "#3b82f6",
    "Transport": "#06b6d4",
    "Entertainment": "#a855f7",
    "EMI": "#ef4444",
    "Transfers": "#64748b",
    "Other": "#94a3b8",
}
CHAT_MONTH_LOOKUP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
CHAT_STOPWORDS = {
    "a", "about", "all", "amount", "and", "any", "are", "as", "by", "can", "credit",
    "credited", "credits", "current", "debit", "debited", "deposited", "deposit",
    "deposits", "did", "do", "done", "expense", "expenses", "for", "from", "give",
    "have", "high", "highest", "how", "i", "in", "is", "largest", "last", "latest",
    "list", "max", "maximum", "me", "mean", "merchant", "merchants", "min", "minimum",
    "money", "month", "monthly", "much", "paid", "payee", "payees", "payment",
    "payments", "received", "repeated", "show", "single", "smallest", "spend", "spent",
    "the", "this", "to", "top", "total", "transaction", "transactions", "was", "what",
    "which", "withdraw", "withdrawal", "withdrawals", "withdrawn",
}
CHAT_SCROLL_SCRIPT = (
    "setTimeout(() => { "
    "const el = document.getElementById('payment-chat-scroll'); "
    "if (el) { el.scrollTop = el.scrollHeight; } "
    "}, 80);"
)

# -------------------------
# Analyzer State
# -------------------------
class AnalyzerState(rx.State):
    # Summary stats
    total_income: str = "₹0.00"
    total_expenses: str = "₹0.00"
    net_balance: str = "₹0.00"

    # Processing flag
    is_processing: bool = False

    # UI data containers
    table_columns: list[str] = ["Date", "Description", "Account/Payee", "Category", "Mode", "Band", "Withdrawal", "Deposit", "Amount", "Anomaly"]
    table_data: list[list[str]] = []
    chart_data: list[dict[str, float]] = []
    monthly_chart_data: list[dict[str, float]] = []
    monthly_stacked_category_data: list[dict[str, float]] = []
    monthly_summary_table: list[list[str]] = []
    monthly_spend_table: list[list[str]] = []
    monthly_spend_chart_data: list[dict[str, float]] = []
    monthly_recurring_table: list[list[str]] = []
    monthly_recurring_chart_data: list[dict[str, float]] = []
    repeated_payment_table: list[list[str]] = []
    repeated_payment_category_chart_data: list[dict[str, float]] = []
    repeated_payment_mode_chart_data: list[dict[str, float]] = []
    repeated_payment_monthly_chart_data: list[dict[str, float]] = []
    detail_table: list[list[str]] = []
    detail_recurring_table: list[list[str]] = []
    top_expenses_table: list[list[str]] = []
    recurring_payments: list[list[str]] = []
    forecast_table: list[list[str]] = []
    category_summary_table: list[list[str]] = []
    category_chart_data: list[dict[str, float]] = []
    amount_band_summary_table: list[list[str]] = []
    amount_band_chart_data: list[dict[str, float]] = []
    micro_category_table: list[list[str]] = []
    micro_category_chart_data: list[dict[str, float]] = []
    micro_leakage_table: list[list[str]] = []
    micro_monthly_chart_data: list[dict[str, float]] = []
    micro_insight_table: list[list[str]] = []

    # Raw transactions for filtering/pagination
    raw_transactions: list[dict] = []

    # Payment assistant
    chat_is_open: bool = False
    chat_input: str = ""
    chat_messages: list[list[str]] = [
        ["assistant", "Ask about payments, deposits, repeated merchants, or monthly totals."]
    ]

    # Filter inputs and pagination
    filter_start: str = ""
    filter_end: str = ""
    page_size: int = 25
    page_number: int = 1
    spend_focus: str = UNDER_LIMIT_FOCUS

    # Filtered totals and page totals
    filtered_total_deposit: str = "₹0.00"
    filtered_total_withdrawal: str = "₹0.00"
    filtered_net: str = "₹0.00"
    page_total_deposit: str = "₹0.00"
    page_total_withdrawal: str = "₹0.00"
    page_net: str = "₹0.00"

    # Recurring totals
    recurring_total: str = "₹0.00"
    repeated_payment_total: str = "₹0.00"
    repeated_payment_count_label: str = "0 payee+amount repeats"
    repeated_payment_top_label: str = "No repeated payee and amount"
    latest_month_label: str = "No month"
    latest_month_total: str = "₹0.00"
    latest_month_under_5k: str = "₹0.00"
    latest_month_5k_to_10k: str = "₹0.00"
    latest_month_recurring: str = "₹0.00"
    recurring_monthly_total: str = "₹0.00"
    recurring_monthly_count_label: str = "0 recurring payments"
    detail_band_filter: str = DETAIL_ALL_BANDS
    detail_scope: str = ""
    detail_key: str = ""
    detail_title: str = "Click a month amount or merchant"
    detail_summary: str = "Transaction dates and amounts will show here."
    detail_total: str = "₹0.00"
    detail_count_label: str = "0 payments"
    micro_expense_total: str = "₹0.00"
    micro_expense_count: str = "0"
    micro_expense_count_label: str = "0 records"
    micro_average_ticket: str = "₹0.00"
    micro_expense_share: str = "0.0%"
    upi_micro_total: str = "₹0.00"
    upi_micro_count: str = "0"
    upi_micro_count_label: str = "0 records"
    restaurant_micro_total: str = "₹0.00"
    restaurant_micro_count: str = "0"
    restaurant_micro_count_label: str = "0 records"

    # Stacked chart controls
    bar_categories: list[str] = DEFAULT_BAR_CATEGORIES.copy()
    top_n_categories: int = 6  # user can change; categories beyond top N grouped into Other

    # Forecast model selection and parameters
    forecast_model: str = "exp_smoothing"  # "linear" or "exp_smoothing"
    exp_smoothing_trend: str = "add"  # "add" or "mul" or "None"
    exp_smoothing_smoothing_level: float = 0.2  # alpha

    # Flag for stacked category data presence
    has_monthly_category_data: bool = False

    # Category mapping rules (tuneable). Format: category -> list of keywords
    CATEGORY_RULES = {
        "Food": [
            "restaurant", "restaurants", "cafe", "coffee", "diner", "canteen", "food", "kitchen",
            "truffles", "bamboo", "curry", "tiffin", "madras", "thatha", "udupi", "zomato",
            "swiggy", "eatfit", "dominos", "pizza", "burger", "kfc", "mcdonald", "starbucks",
            "biryani", "bakery", "sweet", "juice", "chaat", "mess", "bhavan", "a2b",
            "sangeetha", "empire", "meghana", "food court",
        ],
        "Groceries": ["grocery", "groceries", "supermarket", "hyper market", "all day", "zepto", "blinkit", "amazonpaygrocery"],
        "Bills": ["bill", "billpay", "utility", "fastag", "airtel", "jio", "vi ", "bsnl", "bescom", "bwssb", "netflix", "recharge", "airtelpayments"],
        "Transport": ["railways", "metro", "bus", "uber", "ola", "rapido", "taxi", "bm tc", "bmtc", "petrol", "fuel"],
        "Entertainment": ["movie", "spotify", "bookmyshow", "primevideo", "hotstar", "sonyliv", "pink berry"],
        "EMI": ["emi", "chq", "cheque"],
        "Refunds": ["refund", "refunds", "razorpay", "amazon.refunds"],
        "Transfers": ["tpt", "transfer", "imps", "neft", "p2p", "self", "transfer to"],
        "Other": []
    }

    # Last export message (to show success/failure)
    last_export_message: str = ""

    # -------------------------
    # Setter helpers for UI controls (required by rx.select / rx.input on_change)
    # -------------------------
    def set_forecast_model(self, value: str):
        self.forecast_model = value
        try:
            if self.monthly_chart_data:
                df = pd.DataFrame(self.monthly_chart_data)
                df2 = df.rename(columns={'month': 'Month', 'deposit': 'Deposit', 'withdrawal': 'Withdrawal', 'savings': 'Savings'})
                try:
                    df2['MonthPeriod'] = pd.to_datetime(df2['Month'], format='%b %Y', errors='coerce').dt.to_period('M')
                    monthly_df = df2[['MonthPeriod', 'Deposit', 'Withdrawal', 'Savings']].dropna()
                    self._compute_forecast(monthly_df)
                except Exception:
                    self.apply_date_filter()
            else:
                self.apply_date_filter()
        except Exception:
            self.apply_date_filter()

    def set_exp_smoothing_trend(self, value: str):
        self.exp_smoothing_trend = value
        self.apply_date_filter()

    def set_exp_smoothing_smoothing_level(self, value):
        try:
            self.exp_smoothing_smoothing_level = float(value)
        except Exception:
            pass
        self.apply_date_filter()

    def set_top_n_categories(self, value):
        try:
            self.top_n_categories = max(1, int(value))
        except Exception:
            self.top_n_categories = 6
        if self.raw_transactions:
            df_raw = pd.DataFrame(self.raw_transactions)
            if 'Date' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Date'], format="%Y-%m-%d", errors='coerce')
            self.process_transactions(df_raw)

    def set_filter_start(self, value: str):
        self.filter_start = value

    def set_filter_end(self, value: str):
        self.filter_end = value

    def set_page_size(self, value):
        try:
            self.page_size = max(1, int(value))
        except Exception:
            self.page_size = 25
        self.apply_date_filter()

    def set_page_number(self, value):
        try:
            self.page_number = max(1, int(value))
        except Exception:
            self.page_number = 1
        self.apply_date_filter()

    def set_spend_focus(self, value: str):
        self.spend_focus = value if value in SPEND_FOCUS_OPTIONS else UNDER_LIMIT_FOCUS
        self.page_number = 1
        self.apply_date_filter()

    def set_chat_input(self, value: str):
        self.chat_input = value

    def toggle_chat(self):
        self.chat_is_open = not self.chat_is_open

    def close_chat(self):
        self.chat_is_open = False

    def ask_chat(self):
        question = str(self.chat_input or "").strip()
        return self._submit_chat_question(question)

    def ask_chat_form(self, form_data: dict):
        question = str(form_data.get("chat_question", "") or self.chat_input or "").strip()
        return self._submit_chat_question(question)

    def _submit_chat_question(self, question: str):
        if not question:
            return

        current_messages = list(self.chat_messages or [])
        current_messages.append(["user", question])
        current_messages.append(["assistant", self._answer_payment_question(question)])
        self.chat_messages = current_messages[-18:]
        self.chat_input = ""
        self.chat_is_open = True
        return rx.call_script(CHAT_SCROLL_SCRIPT)

    def _chat_dataframe(self) -> pd.DataFrame:
        if not self.raw_transactions:
            return pd.DataFrame()

        df = pd.DataFrame(self.raw_transactions).copy()
        df['DateObj'] = pd.to_datetime(df.get('Date', ''), format="%Y-%m-%d", errors='coerce')
        df['MonthPeriod'] = df['DateObj'].dt.to_period('M')
        for col in ['Withdrawal', 'Deposit', 'Amount']:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        for col in ['Description', 'Merchant', 'PaymentAccountDisplay', 'RepeatPaymentDisplay', 'Category', 'PaymentMode']:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        return df

    def _chat_top_n(self, query: str, default: int = 5) -> int:
        match = re.search(r"\btop\s+(\d{1,2})\b", query.lower())
        if not match:
            return default
        try:
            return min(10, max(1, int(match.group(1))))
        except Exception:
            return default

    def _chat_period_filter(self, df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, str]:
        if df.empty or 'DateObj' not in df.columns:
            return df, "all loaded data"

        valid_dates = df[df['DateObj'].notna()]
        if valid_dates.empty:
            return df, "all loaded data"

        query_lower = query.lower()
        latest_period = valid_dates['MonthPeriod'].max()
        selected_period = None
        label_suffix = ""

        if re.search(r"\b(this|current|latest)\s+month\b", query_lower):
            selected_period = latest_period
            label_suffix = " (latest loaded month)"
        elif re.search(r"\b(last|previous)\s+month\b", query_lower):
            selected_period = latest_period - 1

        if selected_period is None:
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query_lower)
            requested_year = int(year_match.group(1)) if year_match else None
            for month_name, month_number in sorted(CHAT_MONTH_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True):
                if not re.search(rf"\b{month_name}\b", query_lower):
                    continue
                if requested_year:
                    selected_period = pd.Period(year=requested_year, month=month_number, freq='M')
                else:
                    month_matches = valid_dates[valid_dates['DateObj'].dt.month == month_number]
                    if not month_matches.empty:
                        selected_period = month_matches['MonthPeriod'].max()
                break

        if selected_period is None:
            return df, "all loaded data"

        filtered = df[df['MonthPeriod'] == selected_period]
        label = selected_period.to_timestamp().strftime('%b %Y') + label_suffix
        return filtered, label

    def _chat_search_phrase(self, query: str) -> str:
        query_lower = query.lower()
        query_lower = re.sub(r"\b(this|current|latest|last|previous)\s+month\b", " ", query_lower)
        query_lower = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", query_lower)
        for month_name in CHAT_MONTH_LOOKUP:
            query_lower = re.sub(rf"\b{month_name}\b", " ", query_lower)

        candidate = ""
        for pattern in [
            r"\b(?:paid\s+to|payment\s+to|payments\s+to|sent\s+to|for|to|from)\s+([a-z0-9@._&\-\s]{2,})",
            r"\b(?:merchant|payee|source)\s+([a-z0-9@._&\-\s]{2,})",
        ]:
            match = re.search(pattern, query_lower)
            if match:
                candidate = match.group(1)
                break

        if not candidate:
            candidate = query_lower

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", candidate)
            if token not in CHAT_STOPWORDS and not token.isdigit()
        ]
        return " ".join(tokens[:5]).strip()

    def _chat_match_records(self, df: pd.DataFrame, phrase: str) -> pd.DataFrame:
        if df.empty or not phrase:
            return df.iloc[0:0].copy()

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", phrase.lower())
            if token and token not in CHAT_STOPWORDS
        ]
        if not tokens:
            return df.iloc[0:0].copy()

        combined = (
            df['Merchant'] + " " +
            df['PaymentAccountDisplay'] + " " +
            df['RepeatPaymentDisplay'] + " " +
            df['Description']
        )
        normalized = combined.str.lower().str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        mask = normalized.apply(lambda value: all(token in value for token in tokens))
        return df[mask].copy()

    def _chat_group_lines(self, grouped: pd.DataFrame, n: int, amount_label: str = "total") -> list[str]:
        lines = []
        for idx, (_, row) in enumerate(grouped.head(n).iterrows(), start=1):
            lines.append(
                f"{idx}. {row['Merchant']} - {int(row['Count'])} transactions, "
                f"{amount_label} {self._format_currency(float(row['Total']))}"
            )
        return lines

    def _answer_named_total(self, period_df: pd.DataFrame, all_df: pd.DataFrame, query: str, period_label: str, amount_col: str) -> str:
        phrase = self._chat_search_phrase(query)
        movement_label = "deposits from" if amount_col == "Deposit" else "payments to"
        answer_subject = f"Deposits from {phrase.title()}" if amount_col == "Deposit" else f"Payments to {phrase.title()}"
        if not phrase:
            return "Please include a merchant or payee name."

        matches = self._chat_match_records(period_df, phrase)
        matches = matches[matches[amount_col] > 0]
        if matches.empty:
            all_matches = self._chat_match_records(all_df, phrase)
            all_matches = all_matches[all_matches[amount_col] > 0]
            if all_matches.empty:
                return f"No {movement_label} {phrase.title()} found in the loaded statement."
            all_total = float(all_matches[amount_col].sum())
            return (
                f"No {movement_label} {phrase.title()} found in {period_label}.\n"
                f"Across all loaded data: {self._format_currency(all_total)} across {len(all_matches)} transactions."
            )

        total = float(matches[amount_col].sum())
        largest = float(matches[amount_col].max())
        latest_row = matches.sort_values('DateObj').iloc[-1]
        matched_names = ", ".join(matches['Merchant'].value_counts().head(3).index.tolist())
        return (
            f"{answer_subject} in {period_label}: {self._format_currency(total)} "
            f"across {len(matches)} transactions.\n"
            f"- Largest: {self._format_currency(largest)}\n"
            f"- Latest: {latest_row.get('Date', '')}\n"
            f"- Matched: {matched_names or phrase.title()}"
        )

    def _answer_largest_transaction(self, period_df: pd.DataFrame, query: str, period_label: str, amount_col: str) -> str:
        movement_label = "deposit" if amount_col == "Deposit" else "withdrawal"
        phrase = self._chat_search_phrase(query)
        matches = self._chat_match_records(period_df, phrase) if phrase else period_df
        matches = matches[matches[amount_col] > 0]
        if matches.empty:
            return f"No {movement_label}s found in {period_label}."

        transaction = matches.sort_values(amount_col, ascending=False).iloc[0]
        name = transaction.get("Merchant", "") or transaction.get("Description", "") or "Unknown"
        preposition = "from" if amount_col == "Deposit" else "to"
        return (
            f"Highest {movement_label} in {period_label}: "
            f"{self._format_currency(float(transaction[amount_col]))} {preposition} {name} "
            f"on {transaction.get('Date', 'an unknown date')}."
        )

    def _answer_top_repeated_merchants(self, period_df: pd.DataFrame, period_label: str, n: int) -> str:
        withdrawals = period_df[period_df['Withdrawal'] > 0].copy()
        if withdrawals.empty:
            return f"No payments found for {period_label}."

        grouped = (
            withdrawals.groupby('Merchant', dropna=False)
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'))
            .reset_index()
        )
        grouped['Merchant'] = grouped['Merchant'].replace("", "Unknown")
        repeated = grouped[grouped['Count'] >= 2].sort_values(['Count', 'Total'], ascending=[False, False])
        if repeated.empty:
            return f"No repeated merchants found in {period_label}."

        top_total = float(repeated.head(n)['Total'].sum())
        lines = self._chat_group_lines(repeated, n)
        return f"Top {min(n, len(repeated))} repeated merchants in {period_label}: {self._format_currency(top_total)} combined.\n" + "\n".join(lines)

    def _answer_top_payment_merchants(self, period_df: pd.DataFrame, period_label: str, n: int) -> str:
        withdrawals = period_df[period_df['Withdrawal'] > 0].copy()
        if withdrawals.empty:
            return f"No payments found for {period_label}."

        grouped = (
            withdrawals.groupby('Merchant', dropna=False)
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'))
            .reset_index()
            .sort_values(['Total', 'Count'], ascending=[False, False])
        )
        grouped['Merchant'] = grouped['Merchant'].replace("", "Unknown")
        top_total = float(grouped.head(n)['Total'].sum())
        lines = self._chat_group_lines(grouped, n)
        return f"Top {min(n, len(grouped))} payment merchants in {period_label}: {self._format_currency(top_total)} combined.\n" + "\n".join(lines)

    def _answer_top_deposit_sources(self, period_df: pd.DataFrame, period_label: str, n: int) -> str:
        deposits = period_df[period_df['Deposit'] > 0].copy()
        if deposits.empty:
            return f"No deposits found for {period_label}."

        grouped = (
            deposits.groupby('Merchant', dropna=False)
            .agg(Count=('Deposit', 'size'), Total=('Deposit', 'sum'))
            .reset_index()
            .sort_values(['Total', 'Count'], ascending=[False, False])
        )
        grouped['Merchant'] = grouped['Merchant'].replace("", "Unknown")
        top_total = float(grouped.head(n)['Total'].sum())
        lines = self._chat_group_lines(grouped, n)
        return f"Top {min(n, len(grouped))} deposit sources in {period_label}: {self._format_currency(top_total)} combined.\n" + "\n".join(lines)

    def _answer_recurring_merchants(self, period_df: pd.DataFrame, all_df: pd.DataFrame, period_label: str, n: int) -> str:
        all_withdrawals = all_df[all_df['Withdrawal'] > 0].copy()
        if all_withdrawals.empty:
            return "No payments found in the loaded statement."

        recurring_stats = (
            all_withdrawals.groupby('Merchant', dropna=False)
            .agg(Months=('MonthPeriod', 'nunique'))
            .reset_index()
        )
        recurring_merchants = set(recurring_stats.loc[recurring_stats['Months'] >= 2, 'Merchant'])
        if not recurring_merchants:
            return "No merchants paid in two or more months were found."

        selected = period_df[(period_df['Withdrawal'] > 0) & (period_df['Merchant'].isin(recurring_merchants))]
        if selected.empty:
            return f"No recurring merchant payments found in {period_label}."

        grouped = (
            selected.groupby('Merchant', dropna=False)
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'))
            .reset_index()
            .sort_values(['Total', 'Count'], ascending=[False, False])
        )
        grouped['Merchant'] = grouped['Merchant'].replace("", "Unknown")
        total = float(grouped.head(n)['Total'].sum())
        lines = self._chat_group_lines(grouped, n)
        return f"Top {min(n, len(grouped))} recurring merchants in {period_label}: {self._format_currency(total)} combined.\n" + "\n".join(lines)

    def _answer_mode_total(self, period_df: pd.DataFrame, period_label: str, mode: str) -> str:
        records = period_df[(period_df['Withdrawal'] > 0) & (period_df['PaymentMode'] == mode)]
        if records.empty:
            return f"No {mode} payments found in {period_label}."
        total = float(records['Withdrawal'].sum())
        grouped = (
            records.groupby('Merchant', dropna=False)
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'))
            .reset_index()
            .sort_values(['Total', 'Count'], ascending=[False, False])
        )
        lines = self._chat_group_lines(grouped, 5)
        return f"{mode} payments in {period_label}: {self._format_currency(total)} across {len(records)} transactions.\n" + "\n".join(lines)

    def _answer_payment_question(self, query: str) -> str:
        all_df = self._chat_dataframe()
        if all_df.empty:
            return "Upload a statement first, then ask about payments, deposits, merchants, or monthly totals."

        period_df, period_label = self._chat_period_filter(all_df, query)
        query_lower = query.lower()
        top_n = self._chat_top_n(query_lower)

        if any(word in query_lower for word in ["help", "examples", "sample"]):
            return "I can summarize deposits, withdrawals, merchants, repeated payments, payment modes, and monthly totals from the loaded statement."

        if "recurring" in query_lower:
            return self._answer_recurring_merchants(period_df, all_df, period_label, top_n)

        if "repeat" in query_lower or "repeated" in query_lower:
            return self._answer_top_repeated_merchants(period_df, period_label, top_n)

        asks_for_largest = re.search(r"\b(highest|largest|biggest|maximum|max)\b", query_lower)
        if asks_for_largest and any(word in query_lower for word in ["deposit", "deposits", "deposited", "credit", "credits", "credited", "received"]):
            return self._answer_largest_transaction(period_df, query, period_label, "Deposit")
        if asks_for_largest and any(word in query_lower for word in ["withdraw", "withdrawal", "withdrawals", "withdrawn", "debit", "debited", "payment", "payments", "paid", "spend", "spent"]):
            return self._answer_largest_transaction(period_df, query, period_label, "Withdrawal")

        if any(word in query_lower for word in ["deposit", "deposits", "deposited", "credit", "credits", "income", "received"]):
            phrase = self._chat_search_phrase(query)
            if "top" in query_lower or not phrase:
                return self._answer_top_deposit_sources(period_df, period_label, top_n)
            return self._answer_named_total(period_df, all_df, query, period_label, "Deposit")

        mode_lookup = {
            "upi": "UPI",
            "card": "Card",
            "transfer": "Transfer",
            "auto debit": "Auto Debit",
            "autopay": "Auto Debit",
            "atm": "ATM/Cash",
            "cash": "ATM/Cash",
        }
        for token, mode in mode_lookup.items():
            if token in query_lower:
                return self._answer_mode_total(period_df, period_label, mode)

        if "top" in query_lower and any(word in query_lower for word in ["merchant", "merchants", "payee", "payees", "payment", "payments", "expense", "spend"]):
            return self._answer_top_payment_merchants(period_df, period_label, top_n)

        phrase = self._chat_search_phrase(query)
        if phrase:
            return self._answer_named_total(period_df, all_df, query, period_label, "Withdrawal")

        withdrawals = period_df[period_df['Withdrawal'] > 0]
        deposits = period_df[period_df['Deposit'] > 0]
        withdrawal_total = float(withdrawals['Withdrawal'].sum())
        deposit_total = float(deposits['Deposit'].sum())
        return (
            f"Summary for {period_label}:\n"
            f"- Payments: {self._format_currency(withdrawal_total)} across {len(withdrawals)} transactions\n"
            f"- Deposits: {self._format_currency(deposit_total)} across {len(deposits)} transactions\n"
            f"- Net: {self._format_currency(deposit_total - withdrawal_total)}"
        )

    # -------------------------
    # Helpers: persistence
    # -------------------------
    def _load_filter_state(self):
        if os.path.exists(FILTER_STATE_FILE):
            try:
                with open(FILTER_STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.filter_start = data.get("filter_start", "")
                    self.filter_end = data.get("filter_end", "")
                    self.page_size = int(data.get("page_size", self.page_size))
                    self.page_number = int(data.get("page_number", self.page_number))
                    self.spend_focus = data.get("spend_focus", self.spend_focus)
                    if self.spend_focus not in SPEND_FOCUS_OPTIONS:
                        self.spend_focus = UNDER_LIMIT_FOCUS
                    self.top_n_categories = int(data.get("top_n_categories", self.top_n_categories))
                    self.forecast_model = data.get("forecast_model", self.forecast_model)
                    self.exp_smoothing_trend = data.get("exp_smoothing_trend", self.exp_smoothing_trend)
                    self.exp_smoothing_smoothing_level = float(data.get("exp_smoothing_smoothing_level", self.exp_smoothing_smoothing_level))
            except Exception:
                pass

    def _save_filter_state(self):
        try:
            data = {
                "filter_start": self.filter_start,
                "filter_end": self.filter_end,
                "page_size": int(self.page_size),
                "page_number": int(self.page_number),
                "spend_focus": self.spend_focus,
                "top_n_categories": int(self.top_n_categories),
                "forecast_model": self.forecast_model,
                "exp_smoothing_trend": self.exp_smoothing_trend,
                "exp_smoothing_smoothing_level": float(self.exp_smoothing_smoothing_level),
            }
            with open(FILTER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    # -------------------------
    # File upload handler
    # -------------------------
    async def handle_upload(self, files: list[rx.UploadFile]):
        self.is_processing = True
        yield
        for file in files:
            upload_data = await file.read()
            try:
                if file.filename.lower().endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(upload_data), dtype=str)
                elif file.filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(upload_data), dtype=str)
                else:
                    print(f"Skipping unsupported file type: {file.filename}")
                    continue
                self.process_transactions(df)
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
        self.is_processing = False

    # -------------------------
    # Category rules upload (CSV: keyword,category)
    # -------------------------
    async def handle_category_rules_upload(self, files: list[rx.UploadFile]):
        """Accept a small CSV with columns 'keyword,category' to update CATEGORY_RULES."""
        for file in files:
            try:
                data = await file.read()
                df = pd.read_csv(io.BytesIO(data), dtype=str, header=None)
                new_rules = {}
                for _, row in df.iterrows():
                    if len(row) < 2:
                        continue
                    kw = str(row[0]).strip().lower()
                    cat = str(row[1]).strip()
                    if not kw or not cat:
                        continue
                    new_rules.setdefault(cat, []).append(kw)
                for cat, kws in new_rules.items():
                    if cat in self.CATEGORY_RULES:
                        existing = set(self.CATEGORY_RULES[cat])
                        existing.update(kws)
                        self.CATEGORY_RULES[cat] = list(existing)
                    else:
                        self.CATEGORY_RULES[cat] = kws
                if self.raw_transactions:
                    df_raw = pd.DataFrame(self.raw_transactions)
                    if 'Date' in df_raw.columns:
                        df_raw['Date'] = pd.to_datetime(df_raw['Date'], format="%Y-%m-%d", errors='coerce')
                    self.process_transactions(df_raw)
            except Exception as e:
                print("Failed to load category rules:", e)

    # -------------------------
    # Column helpers
    # -------------------------
    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _parse_statement_dates(self, series: pd.Series) -> pd.Series:
        raw = series.astype(str).str.strip()
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d-%b-%Y", "%d %B %Y"):
            missing = parsed.isna()
            if not missing.any():
                break
            parsed.loc[missing] = pd.to_datetime(raw.loc[missing], format=date_format, errors='coerce')

        missing = parsed.isna()
        if missing.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed.loc[missing] = pd.to_datetime(raw.loc[missing], dayfirst=True, errors='coerce')
        return parsed

    def _assign_category(self, description: str) -> str:
        if not isinstance(description, str):
            return "Other"
        desc = description.lower()
        for cat, keywords in self.CATEGORY_RULES.items():
            for kw in keywords:
                if kw in desc:
                    return cat
        return "Other"

    def _format_currency(self, value: float) -> str:
        return f"₹{float(value or 0.0):,.2f}"

    def _format_percent(self, numerator: float, denominator: float) -> str:
        if not denominator:
            return "0.0%"
        return f"{(float(numerator or 0.0) / float(denominator)) * 100:.1f}%"

    def _month_label_from_date(self, date_value: str) -> str:
        dt = pd.to_datetime(date_value, format="%Y-%m-%d", errors='coerce')
        if pd.isna(dt):
            return ""
        return dt.strftime('%b %Y')

    def _detect_payment_mode(self, description: str) -> str:
        if not isinstance(description, str):
            return "Other"
        desc = description.lower()
        if any(token in desc for token in ["upi", "gpay", "google pay", "phonepe", "paytm", "bhim", "@upi", "p2m", "p2a", "vpa"]):
            return "UPI"
        if any(token in desc for token in ["pos", "debit card", "credit card", "card", "visa", "mastercard", "rupay", "swipe"]):
            return "Card"
        if any(token in desc for token in ["imps", "neft", "rtgs", "tpt", "transfer"]):
            return "Transfer"
        if any(token in desc for token in ["ach", "nach", "ecs", "autopay", "standing instruction", "si "]):
            return "Auto Debit"
        if any(token in desc for token in ["atm", "cash withdrawal", "cash wdl"]):
            return "ATM/Cash"
        return "Other"

    def _spend_band(self, withdrawal: float) -> str:
        try:
            amount = float(withdrawal or 0.0)
        except Exception:
            amount = 0.0
        if amount <= 0:
            return INCOME_BAND
        if amount < MICRO_LIMIT:
            return MICRO_BAND
        if amount < CORE_LIMIT:
            return CORE_BAND
        return LARGE_BAND

    def _is_under_analysis_limit(self, withdrawal: float, deposit: float, amount: float) -> bool:
        try:
            movement = max(abs(float(withdrawal or 0.0)), abs(float(deposit or 0.0)), abs(float(amount or 0.0)))
        except Exception:
            movement = 0.0
        return 0 < movement < ANALYSIS_AMOUNT_LIMIT

    def _record_matches_detail_band(self, rec: dict, band_filter: str) -> bool:
        withdrawal = float(rec.get('Withdrawal', 0.0) or 0.0)
        if withdrawal <= 0:
            return False
        if band_filter == DETAIL_UNDER_5K:
            return withdrawal < MICRO_LIMIT
        if band_filter == DETAIL_5K_TO_10K:
            return MICRO_LIMIT <= withdrawal < CORE_LIMIT
        return withdrawal < CORE_LIMIT

    def _set_detail_records(self, records: list[dict], title: str, summary_prefix: str):
        filtered_records = [
            rec for rec in records
            if self._record_matches_detail_band(rec, self.detail_band_filter)
        ]
        filtered_records.sort(key=lambda rec: rec.get('Date', ''), reverse=True)

        total = sum(float(rec.get('Withdrawal', 0.0) or 0.0) for rec in filtered_records)
        count = len(filtered_records)
        self.detail_title = title
        self.detail_total = self._format_currency(total)
        self.detail_count_label = f"{count} payments"
        self.detail_summary = f"{summary_prefix} · {self.detail_band_filter}"
        self.detail_table = [
            [
                rec.get('Date', ''),
                rec.get('PaymentAccountDisplay', rec.get('Merchant', '')),
                rec.get('Merchant', ''),
                rec.get('Description', ''),
                rec.get('Category', ''),
                rec.get('PaymentMode', ''),
                rec.get('SpendBand', ''),
                self._format_currency(float(rec.get('Withdrawal', 0.0) or 0.0)),
            ]
            for rec in filtered_records
        ]
        self._set_detail_recurring_records(filtered_records)

    def _set_detail_recurring_records(self, selected_records: list[dict]):
        if self.detail_scope != "month" or not self.raw_transactions:
            self.detail_recurring_table = []
            return

        merchant_stats = {}
        for rec in self.raw_transactions:
            if not self._record_matches_detail_band(rec, self.detail_band_filter):
                continue
            merchant = rec.get('Merchant', '') or "Unknown"
            stats = merchant_stats.setdefault(merchant, {"months": set(), "total": 0.0})
            stats["months"].add(self._month_label_from_date(rec.get('Date', '')))
            stats["total"] += float(rec.get('Withdrawal', 0.0) or 0.0)

        month_rows = {}
        for rec in selected_records:
            merchant = rec.get('Merchant', '') or "Unknown"
            stats = merchant_stats.get(merchant, {"months": set(), "total": 0.0})
            if len(stats["months"]) < 2:
                continue
            row = month_rows.setdefault(
                merchant,
                {
                    "merchant": merchant,
                    "category": rec.get('Category', ''),
                    "mode": rec.get('PaymentMode', ''),
                    "count": 0,
                    "month_total": 0.0,
                    "all_total": float(stats["total"]),
                    "months_paid": len(stats["months"]),
                    "last_date": rec.get('Date', ''),
                },
            )
            row["count"] += 1
            row["month_total"] += float(rec.get('Withdrawal', 0.0) or 0.0)
            row["last_date"] = max(str(row["last_date"]), str(rec.get('Date', '')))

        rows = sorted(month_rows.values(), key=lambda row: row["month_total"], reverse=True)
        self.detail_recurring_table = [
            [
                row["merchant"],
                row["category"],
                row["mode"],
                self._format_currency(row["month_total"]),
                str(row["count"]),
                str(row["months_paid"]),
                self._format_currency(row["all_total"]),
                row["last_date"],
            ]
            for row in rows
        ]

    def _refresh_detail_selection(self):
        if not self.raw_transactions:
            self.detail_table = []
            self.detail_recurring_table = []
            self.detail_total = "₹0.00"
            self.detail_count_label = "0 payments"
            return

        if self.detail_scope == "month":
            month = self.detail_key
            records = [
                rec for rec in self.raw_transactions
                if self._month_label_from_date(rec.get('Date', '')) == month
            ]
            self._set_detail_records(records, f"{month} transactions", "Monthly spend")
            return

        if self.detail_scope == "merchant":
            merchant = self.detail_key
            records = [
                rec for rec in self.raw_transactions
                if rec.get('Merchant', '') == merchant
            ]
            self._set_detail_records(records, merchant, "Merchant payment history")
            return

        if self.detail_scope == "repeat":
            repeat_key = self.detail_key
            records = [
                rec for rec in self.raw_transactions
                if rec.get('RepeatPaymentKey', '') == repeat_key
            ]
            if records:
                account = (
                    records[0].get('RepeatPaymentDisplay', '')
                    or records[0].get('PaymentAccountDisplay', '')
                    or records[0].get('Merchant', '')
                    or "Repeated payment"
                )
                amount = self._format_currency(float(records[0].get('Withdrawal', 0.0) or 0.0))
                self._set_detail_records(records, f"{account} · {amount}", "Same payee and amount")
            else:
                self.detail_table = []
                self.detail_recurring_table = []
                self.detail_total = "₹0.00"
                self.detail_count_label = "0 payments"
                self.detail_title = "Repeated payment not found"
                self.detail_summary = "No matching records are loaded."
            return

        self.detail_table = []
        self.detail_recurring_table = []
        self.detail_total = "₹0.00"
        self.detail_count_label = "0 payments"

    def set_detail_band_filter(self, value: str):
        self.detail_band_filter = value if value in DETAIL_BAND_FILTER_OPTIONS else DETAIL_ALL_BANDS
        self._refresh_detail_selection()
        if self.detail_scope:
            return rx.redirect("/details")

    def show_month_details(self, month: str, band_filter: str = DETAIL_ALL_BANDS):
        self.detail_scope = "month"
        self.detail_key = str(month)
        self.detail_band_filter = band_filter if band_filter in DETAIL_BAND_FILTER_OPTIONS else DETAIL_ALL_BANDS
        self._refresh_detail_selection()
        return rx.redirect("/details")

    def show_merchant_details(self, merchant: str):
        self.detail_scope = "merchant"
        self.detail_key = str(merchant)
        self.detail_band_filter = DETAIL_ALL_BANDS
        self._refresh_detail_selection()
        return rx.redirect("/details")

    def show_repeated_payment_details(self, repeat_key: str):
        self.detail_scope = "repeat"
        self.detail_key = str(repeat_key)
        self.detail_band_filter = DETAIL_ALL_BANDS
        self._refresh_detail_selection()
        return rx.redirect("/details")

    def _merchant_name(self, description: str) -> str:
        if not isinstance(description, str) or not description.strip():
            return "Unknown"
        desc = description.lower()
        desc = re.sub(r"\b\d{4,}\b", " ", desc)
        desc = re.sub(r"[^a-z0-9 ]+", " ", desc)
        desc = re.sub(
            r"\b(upi|payment|pay|paid|txn|ref|rrn|imps|neft|rtgs|transfer|from|to|by|"
            r"p2a|p2m|collect|debit|credit|card|pos|at|on|id|vpa|ybl|ibl|oksbi|"
            r"okaxis|okhdfcbank|okicici|paytm|phonepe|gpay|google)\b",
            " ",
            desc,
        )
        desc = re.sub(r"\s+", " ", desc).strip()
        if not desc:
            fallback = re.sub(r"[^a-z0-9 ]+", " ", description.lower())
            desc = re.sub(r"\s+", " ", fallback).strip()
        return (desc[:45] or "Unknown").title()

    def _mask_identifier(self, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return "Unknown"
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 6:
            return f"...{digits[-4:]}"
        return value[:28]

    def _payment_account_parts(self, description: str, merchant: str) -> tuple[str, str]:
        raw_description = str(description or "")
        desc = raw_description.lower()

        vpa_match = re.search(r"\b[a-z0-9][a-z0-9._%+\-]{1,}@[a-z0-9.\-]{2,}\b", desc)
        if vpa_match:
            vpa = vpa_match.group(0)
            return f"upi:{vpa}", f"UPI {vpa}"

        account_patterns = [
            r"(?:a/c|ac|acct|account|acc)\s*(?:no\.?|number|num)?\s*[:\-/]?\s*([xX*]*\d[\dxX*]{4,})",
            r"(?:beneficiary|bene|to account|from account)\s*[:\-/]?\s*([xX*]*\d[\dxX*]{4,})",
        ]
        for pattern in account_patterns:
            account_match = re.search(pattern, desc)
            if account_match:
                account_value = re.sub(r"[^a-z0-9x*]", "", account_match.group(1).lower())
                return f"acct:{account_value}", f"Account {self._mask_identifier(account_value)}"

        merchant_value = str(merchant or "").strip() or "Unknown"
        merchant_key = re.sub(r"[^a-z0-9]+", " ", merchant_value.lower()).strip() or "unknown"
        return f"merchant:{merchant_key}", merchant_value[:55]

    def _repeat_payee_parts(self, merchant: str) -> tuple[str, str]:
        merchant_value = str(merchant or "").strip()
        words = re.findall(r"[a-z0-9]+", merchant_value.lower())
        if not words:
            return "payee:unknown", "Unknown"
        first_two_words = words[:2]
        payee_key = " ".join(first_two_words)
        payee_display = " ".join(first_two_words).title()
        return f"payee:{payee_key}", payee_display

    def _repeat_payment_key(self, account_key: str, withdrawal: float) -> str:
        try:
            amount = float(withdrawal or 0.0)
        except Exception:
            amount = 0.0
        if amount <= 0:
            return ""
        return f"{account_key}|{amount:.2f}"

    def _record_matches_focus(self, rec: dict) -> bool:
        focus = self.spend_focus if self.spend_focus in SPEND_FOCUS_OPTIONS else UNDER_LIMIT_FOCUS
        if not self._is_under_analysis_limit(
            rec.get('Withdrawal', 0.0),
            rec.get('Deposit', 0.0),
            rec.get('Amount', 0.0),
        ):
            return False
        if focus == UNDER_LIMIT_FOCUS:
            return True

        withdrawal = float(rec.get('Withdrawal', 0.0) or 0.0)
        if withdrawal <= 0:
            return False

        band = rec.get('SpendBand') or self._spend_band(withdrawal)
        mode = rec.get('PaymentMode', '')
        category = rec.get('Category', '')

        if focus == MICRO_BAND:
            return band == MICRO_BAND
        if focus == CORE_BAND:
            return band == CORE_BAND
        if focus == "UPI micro":
            return band == MICRO_BAND and mode == "UPI"
        if focus == "Food & restaurants micro":
            return band == MICRO_BAND and category == "Food"
        return True

    def _reset_micro_analysis(self):
        self.amount_band_summary_table = []
        self.amount_band_chart_data = []
        self.micro_category_table = []
        self.micro_category_chart_data = []
        self.micro_leakage_table = []
        self.micro_monthly_chart_data = []
        self.micro_insight_table = []
        self.micro_expense_total = "₹0.00"
        self.micro_expense_count = "0"
        self.micro_expense_count_label = "0 records"
        self.micro_average_ticket = "₹0.00"
        self.micro_expense_share = "0.0%"
        self.upi_micro_total = "₹0.00"
        self.upi_micro_count = "0"
        self.upi_micro_count_label = "0 records"
        self.restaurant_micro_total = "₹0.00"
        self.restaurant_micro_count = "0"
        self.restaurant_micro_count_label = "0 records"

    def _reset_repeated_payments(self):
        self.repeated_payment_table = []
        self.repeated_payment_category_chart_data = []
        self.repeated_payment_mode_chart_data = []
        self.repeated_payment_monthly_chart_data = []
        self.repeated_payment_total = "₹0.00"
        self.repeated_payment_count_label = "0 payee+amount repeats"
        self.repeated_payment_top_label = "No repeated payee and amount"

    def _clear_analysis_results(self):
        self.total_income = "₹0.00"
        self.total_expenses = "₹0.00"
        self.net_balance = "₹0.00"
        self.table_data = []
        self.chart_data = []
        self.monthly_chart_data = []
        self.monthly_summary_table = []
        self.monthly_spend_table = []
        self.monthly_spend_chart_data = []
        self.monthly_recurring_table = []
        self.monthly_recurring_chart_data = []
        self._reset_repeated_payments()
        self.detail_table = []
        self.detail_recurring_table = []
        self.top_expenses_table = []
        self.recurring_payments = []
        self.forecast_table = []
        self.category_summary_table = []
        self.category_chart_data = []
        self.monthly_stacked_category_data = []
        self.raw_transactions = []
        self.has_monthly_category_data = False
        self.filtered_total_deposit = "₹0.00"
        self.filtered_total_withdrawal = "₹0.00"
        self.filtered_net = "₹0.00"
        self.page_total_deposit = "₹0.00"
        self.page_total_withdrawal = "₹0.00"
        self.page_net = "₹0.00"
        self.recurring_total = "₹0.00"
        self.latest_month_label = "No month"
        self.latest_month_total = "₹0.00"
        self.latest_month_under_5k = "₹0.00"
        self.latest_month_5k_to_10k = "₹0.00"
        self.latest_month_recurring = "₹0.00"
        self.recurring_monthly_total = "₹0.00"
        self.recurring_monthly_count_label = "0 recurring payments"
        self.detail_band_filter = DETAIL_ALL_BANDS
        self.detail_scope = ""
        self.detail_key = ""
        self.detail_title = "Click a month amount or merchant"
        self.detail_summary = "Transaction dates and amounts will show here."
        self.detail_total = "₹0.00"
        self.detail_count_label = "0 payments"
        self._reset_micro_analysis()

    def _reset_monthly_essentials(self):
        self.monthly_spend_table = []
        self.monthly_spend_chart_data = []
        self.monthly_recurring_table = []
        self.monthly_recurring_chart_data = []
        self.detail_table = []
        self.detail_recurring_table = []
        self.latest_month_label = "No month"
        self.latest_month_total = "₹0.00"
        self.latest_month_under_5k = "₹0.00"
        self.latest_month_5k_to_10k = "₹0.00"
        self.latest_month_recurring = "₹0.00"
        self.recurring_monthly_total = "₹0.00"
        self.recurring_monthly_count_label = "0 recurring payments"
        self.detail_scope = ""
        self.detail_key = ""
        self.detail_title = "Click a month amount or merchant"
        self.detail_summary = "Transaction dates and amounts will show here."
        self.detail_total = "₹0.00"
        self.detail_count_label = "0 payments"

    def _compute_monthly_essentials(self, df: pd.DataFrame):
        withdrawals = df[df['Withdrawal'] > 0].copy()
        if withdrawals.empty:
            self._reset_monthly_essentials()
            return

        self.detail_scope = ""
        self.detail_key = ""
        self.detail_table = []
        self.detail_total = "₹0.00"
        self.detail_count_label = "0 payments"
        self.detail_title = "Click a month amount or merchant"
        self.detail_summary = "Transaction dates and amounts will show here."

        withdrawals['MonthPeriod'] = withdrawals['Date'].dt.to_period('M')
        monthly_rows = []
        monthly_chart = []
        for month_period, month_df in withdrawals.groupby('MonthPeriod', sort=True):
            under_5k = float(month_df.loc[month_df['Withdrawal'] < MICRO_LIMIT, 'Withdrawal'].sum())
            five_to_10k = float(month_df.loc[
                (month_df['Withdrawal'] >= MICRO_LIMIT) & (month_df['Withdrawal'] < CORE_LIMIT),
                'Withdrawal',
            ].sum())
            total = under_5k + five_to_10k
            count = int(len(month_df))
            month_label = month_period.to_timestamp().strftime('%b %Y')
            monthly_rows.append([
                month_label,
                self._format_currency(under_5k),
                self._format_currency(five_to_10k),
                self._format_currency(total),
                str(count),
            ])
            monthly_chart.append({
                "Month": month_label,
                "Under5k": under_5k,
                "FiveToTenK": five_to_10k,
                "Total": total,
            })

        self.monthly_spend_table = monthly_rows
        self.monthly_spend_chart_data = monthly_chart
        latest = monthly_chart[-1]
        self.latest_month_label = str(latest["Month"])
        self.latest_month_under_5k = self._format_currency(float(latest["Under5k"]))
        self.latest_month_5k_to_10k = self._format_currency(float(latest["FiveToTenK"]))
        self.latest_month_total = self._format_currency(float(latest["Total"]))

        recurring_rows = []
        recurring_keys = set()
        total_months = max(1, withdrawals['MonthPeriod'].nunique())
        merchant_group = withdrawals.groupby(['Merchant', 'Category', 'PaymentMode'], dropna=False)
        for (merchant, category, mode), group in merchant_group:
            monthly_group = group.groupby('MonthPeriod')['Withdrawal'].sum().sort_index()
            months_paid = int(len(monthly_group))
            if months_paid < 2:
                continue
            total_paid = float(monthly_group.sum())
            avg_monthly = float(monthly_group.mean())
            min_monthly = float(monthly_group.min())
            max_monthly = float(monthly_group.max())
            latest_month = monthly_group.index[-1].to_timestamp().strftime('%b %Y')
            regularity = f"{months_paid}/{total_months} months"
            amount_range = (
                self._format_currency(min_monthly)
                if round(min_monthly, 2) == round(max_monthly, 2)
                else f"{self._format_currency(min_monthly)} - {self._format_currency(max_monthly)}"
            )
            recurring_rows.append([
                merchant,
                category,
                mode,
                regularity,
                self._format_currency(avg_monthly),
                self._format_currency(total_paid),
                amount_range,
                latest_month,
                months_paid,
                total_paid,
            ])
            recurring_keys.add((merchant, category, mode))

        recurring_rows.sort(key=lambda row: (row[8], row[9]), reverse=True)
        self.monthly_recurring_table = [row[:8] for row in recurring_rows[:20]]
        self.recurring_monthly_total = self._format_currency(sum(float(row[9]) for row in recurring_rows))
        self.recurring_monthly_count_label = f"{len(recurring_rows)} recurring payments"

        if recurring_keys:
            recurring_df = withdrawals[
                withdrawals[['Merchant', 'Category', 'PaymentMode']]
                .apply(tuple, axis=1)
                .isin(recurring_keys)
            ].copy()
            latest_period = withdrawals['MonthPeriod'].max()
            latest_recurring = float(recurring_df.loc[recurring_df['MonthPeriod'] == latest_period, 'Withdrawal'].sum())
            self.latest_month_recurring = self._format_currency(latest_recurring)
            recurring_monthly = (
                recurring_df.groupby('MonthPeriod')
                .agg(RecurringSpend=('Withdrawal', 'sum'), Payments=('Withdrawal', 'size'))
                .reset_index()
                .sort_values('MonthPeriod')
            )
            recurring_monthly['Month'] = recurring_monthly['MonthPeriod'].dt.to_timestamp().dt.strftime('%b %Y')
            self.monthly_recurring_chart_data = recurring_monthly[['Month', 'RecurringSpend', 'Payments']].to_dict('records')
        else:
            self.latest_month_recurring = "₹0.00"
            self.monthly_recurring_chart_data = []

    def _compute_micro_leakage(self, df: pd.DataFrame, total_withdrawals: float):
        withdrawals = df[df['Withdrawal'] > 0].copy()
        if withdrawals.empty:
            self._reset_micro_analysis()
            return

        if 'PaymentMode' not in withdrawals.columns:
            withdrawals['PaymentMode'] = withdrawals['Description'].apply(self._detect_payment_mode)
        if 'SpendBand' not in withdrawals.columns:
            withdrawals['SpendBand'] = withdrawals['Withdrawal'].apply(self._spend_band)
        if 'Merchant' not in withdrawals.columns:
            withdrawals['Merchant'] = withdrawals['Description'].apply(self._merchant_name)

        band_chart = []
        band_rows = []
        for band in [MICRO_BAND, CORE_BAND]:
            band_df = withdrawals[withdrawals['SpendBand'] == band]
            total = float(band_df['Withdrawal'].sum())
            count = int(len(band_df))
            avg = float(band_df['Withdrawal'].mean()) if count else 0.0
            upi_total = float(band_df.loc[band_df['PaymentMode'] == "UPI", 'Withdrawal'].sum())
            food_total = float(band_df.loc[band_df['Category'] == "Food", 'Withdrawal'].sum())
            band_rows.append([
                band,
                str(count),
                self._format_currency(total),
                self._format_currency(avg),
                self._format_percent(total, total_withdrawals),
                self._format_currency(upi_total),
                self._format_currency(food_total),
            ])
            band_chart.append({"name": band, "value": total})
        self.amount_band_summary_table = band_rows
        self.amount_band_chart_data = band_chart

        micro_df = withdrawals[withdrawals['SpendBand'] == MICRO_BAND].copy()
        micro_count = int(len(micro_df))
        micro_total = float(micro_df['Withdrawal'].sum())
        micro_avg = float(micro_df['Withdrawal'].mean()) if micro_count else 0.0
        upi_micro_df = micro_df[micro_df['PaymentMode'] == "UPI"]
        food_micro_df = micro_df[micro_df['Category'] == "Food"]

        self.micro_expense_total = self._format_currency(micro_total)
        self.micro_expense_count = str(micro_count)
        self.micro_expense_count_label = f"{micro_count} records"
        self.micro_average_ticket = self._format_currency(micro_avg)
        self.micro_expense_share = self._format_percent(micro_total, total_withdrawals)
        self.upi_micro_total = self._format_currency(float(upi_micro_df['Withdrawal'].sum()))
        self.upi_micro_count = str(int(len(upi_micro_df)))
        self.upi_micro_count_label = f"{int(len(upi_micro_df))} records"
        self.restaurant_micro_total = self._format_currency(float(food_micro_df['Withdrawal'].sum()))
        self.restaurant_micro_count = str(int(len(food_micro_df)))
        self.restaurant_micro_count_label = f"{int(len(food_micro_df))} records"

        if micro_df.empty:
            self.micro_category_table = []
            self.micro_category_chart_data = []
            self.micro_leakage_table = []
            self.micro_monthly_chart_data = []
            self.micro_insight_table = [["Micro leakage", "₹0.00", "No sub-₹5,000 withdrawals found"]]
            return

        category_group = (
            micro_df.groupby('Category')
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'), Average=('Withdrawal', 'mean'))
            .reset_index()
            .sort_values('Total', ascending=False)
        )
        self.micro_category_table = [
            [
                row['Category'],
                str(int(row['Count'])),
                self._format_currency(float(row['Total'])),
                self._format_currency(float(row['Average'])),
                self._format_percent(float(row['Total']), micro_total),
            ]
            for _, row in category_group.head(10).iterrows()
        ]
        self.micro_category_chart_data = [
            {"name": row['Category'], "value": float(row['Total'])}
            for _, row in category_group.head(8).iterrows()
        ]

        merchant_group = (
            micro_df.groupby(['Merchant', 'Category', 'PaymentMode'])
            .agg(Count=('Withdrawal', 'size'), Total=('Withdrawal', 'sum'), Average=('Withdrawal', 'mean'), Largest=('Withdrawal', 'max'))
            .reset_index()
            .sort_values(['Total', 'Count'], ascending=[False, False])
        )
        repeat_group = merchant_group[merchant_group['Count'] >= 2]
        if repeat_group.empty:
            repeat_group = merchant_group
        self.micro_leakage_table = [
            [
                row['Merchant'],
                row['Category'],
                row['PaymentMode'],
                str(int(row['Count'])),
                self._format_currency(float(row['Total'])),
                self._format_currency(float(row['Average'])),
                self._format_currency(float(row['Largest'])),
            ]
            for _, row in repeat_group.head(12).iterrows()
        ]

        monthly_micro = (
            micro_df.groupby('MonthPeriod')
            .agg(MicroSpend=('Withdrawal', 'sum'), Count=('Withdrawal', 'size'))
            .reset_index()
            .sort_values('MonthPeriod')
        )
        upi_month = micro_df[micro_df['PaymentMode'] == "UPI"].groupby('MonthPeriod')['Withdrawal'].sum()
        food_month = micro_df[micro_df['Category'] == "Food"].groupby('MonthPeriod')['Withdrawal'].sum()
        monthly_micro['UPI'] = monthly_micro['MonthPeriod'].map(upi_month).fillna(0.0)
        monthly_micro['Food'] = monthly_micro['MonthPeriod'].map(food_month).fillna(0.0)
        monthly_micro['Month'] = monthly_micro['MonthPeriod'].dt.to_timestamp().dt.strftime('%b %Y')
        self.micro_monthly_chart_data = monthly_micro[['Month', 'MicroSpend', 'UPI', 'Food']].to_dict('records')

        top_category = category_group.iloc[0]
        top_merchant = merchant_group.iloc[0]
        upi_total = float(upi_micro_df['Withdrawal'].sum())
        food_total = float(food_micro_df['Withdrawal'].sum())
        self.micro_insight_table = [
            ["Micro share", self.micro_expense_share, f"{micro_count} records below ₹5,000"],
            ["Top micro category", str(top_category['Category']), self._format_currency(float(top_category['Total']))],
            ["Top repeat merchant", str(top_merchant['Merchant']), self._format_currency(float(top_merchant['Total']))],
            ["UPI micro", self._format_percent(upi_total, micro_total), self._format_currency(upi_total)],
            ["Food/restaurant micro", self._format_percent(food_total, micro_total), self._format_currency(food_total)],
        ]

    def _compute_repeated_payments(self, df: pd.DataFrame):
        withdrawals = df[df['Withdrawal'] > 0].copy()
        if withdrawals.empty:
            self._reset_repeated_payments()
            return

        required_cols = {"RepeatPaymentKey", "RepeatPaymentDisplay", "MonthPeriod"}
        if not required_cols.issubset(set(withdrawals.columns)):
            self._reset_repeated_payments()
            return

        rows = []
        repeated_keys = set()
        for repeat_key, group in withdrawals.groupby('RepeatPaymentKey', dropna=False):
            if not repeat_key or len(group) < 2:
                continue
            group = group.sort_values('Date')
            count = int(len(group))
            total_paid = float(group['Withdrawal'].sum())
            amount = float(group['Withdrawal'].iloc[0])
            account_display = str(group['RepeatPaymentDisplay'].iloc[0] or "Unknown")
            category = str(group['Category'].mode().iloc[0]) if not group['Category'].mode().empty else str(group['Category'].iloc[0])
            mode = str(group['PaymentMode'].mode().iloc[0]) if not group['PaymentMode'].mode().empty else str(group['PaymentMode'].iloc[0])
            months_paid = int(group['MonthPeriod'].nunique())
            first_date = group['Date'].min().strftime('%Y-%m-%d')
            last_date = group['Date'].max().strftime('%Y-%m-%d')
            rows.append([
                repeat_key,
                account_display,
                self._format_currency(amount),
                str(count),
                self._format_currency(total_paid),
                category,
                mode,
                str(months_paid),
                first_date,
                last_date,
                count,
                total_paid,
            ])
            repeated_keys.add(repeat_key)

        if not rows:
            self._reset_repeated_payments()
            return

        rows.sort(key=lambda row: (row[10], row[11]), reverse=True)
        self.repeated_payment_table = [row[:10] for row in rows[:30]]
        self.repeated_payment_total = self._format_currency(sum(float(row[11]) for row in rows))
        self.repeated_payment_count_label = f"{len(rows)} payee+amount repeats"
        self.repeated_payment_top_label = f"{rows[0][1]} · {rows[0][2]} · {rows[0][3]} times"

        repeated_df = withdrawals[withdrawals['RepeatPaymentKey'].isin(repeated_keys)].copy()

        category_totals = (
            repeated_df.groupby('Category')
            .agg(Total=('Withdrawal', 'sum'), Count=('Withdrawal', 'size'))
            .reset_index()
            .sort_values('Total', ascending=False)
        )
        self.repeated_payment_category_chart_data = [
            {"Category": str(row['Category']), "Total": float(row['Total']), "Count": int(row['Count'])}
            for _, row in category_totals.iterrows()
        ]

        mode_totals = (
            repeated_df.groupby('PaymentMode')
            .agg(Total=('Withdrawal', 'sum'), Count=('Withdrawal', 'size'))
            .reset_index()
            .sort_values('Total', ascending=False)
        )
        self.repeated_payment_mode_chart_data = [
            {"name": str(row['PaymentMode']), "value": float(row['Total']), "Count": int(row['Count'])}
            for _, row in mode_totals.iterrows()
        ]

        monthly_repeats = (
            repeated_df.groupby('MonthPeriod')
            .agg(RepeatedSpend=('Withdrawal', 'sum'), Payments=('Withdrawal', 'size'))
            .reset_index()
            .sort_values('MonthPeriod')
        )
        monthly_repeats['Month'] = monthly_repeats['MonthPeriod'].dt.to_timestamp().dt.strftime('%b %Y')
        self.repeated_payment_monthly_chart_data = monthly_repeats[['Month', 'RepeatedSpend', 'Payments']].to_dict('records')

    # -------------------------
    # Export helpers (safe server-side action + UI message)
    # -------------------------
    def _export_to_csv_file(self, df: pd.DataFrame, prefix: str) -> Optional[str]:
        try:
            fname = os.path.join(EXPORT_FOLDER, f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(fname, index=False, encoding='utf-8')
            return fname
        except Exception as e:
            print("Export failed:", e)
            return None

    def do_export_filtered_transactions(self):
        """Server action called by button. Writes CSV and sets last_export_message."""
        try:
            if not self.table_data:
                self.last_export_message = "No filtered transactions to export."
                return
            df = pd.DataFrame(self.table_data, columns=self.table_columns)
            path = self._export_to_csv_file(df, "filtered_transactions")
            if path:
                self.last_export_message = f"Exported filtered transactions to: {path}"
            else:
                self.last_export_message = "Export failed (see server logs)."
        except Exception as e:
            print("Export error:", e)
            self.last_export_message = "Export failed (exception)."

    def do_export_monthly_summary(self):
        """Server action called by button. Writes CSV and sets last_export_message."""
        try:
            if not self.monthly_summary_table:
                self.last_export_message = "No monthly summary to export."
                return
            df = pd.DataFrame(self.monthly_summary_table, columns=["Month", "Deposit", "Withdrawal", "Savings"])
            path = self._export_to_csv_file(df, "monthly_summary")
            if path:
                self.last_export_message = f"Exported monthly summary to: {path}"
            else:
                self.last_export_message = "Export failed (see server logs)."
        except Exception as e:
            print("Export error:", e)
            self.last_export_message = "Export failed (exception)."

    # -------------------------
    # Filtering / Pagination
    # -------------------------
    def apply_date_filter(self):
        # persist state
        self._save_filter_state()

        if not self.raw_transactions:
            self.table_data = []
            self.filtered_total_deposit = "₹0.00"
            self.filtered_total_withdrawal = "₹0.00"
            self.filtered_net = "₹0.00"
            self.page_total_deposit = "₹0.00"
            self.page_total_withdrawal = "₹0.00"
            self.page_net = "₹0.00"
            return

        start = pd.to_datetime(self.filter_start, format="%Y-%m-%d", errors='coerce') if self.filter_start else None
        end = pd.to_datetime(self.filter_end, format="%Y-%m-%d", errors='coerce') if self.filter_end else None

        filtered = []
        for rec in self.raw_transactions:
            rec_date = pd.to_datetime(rec.get('Date'), format="%Y-%m-%d", errors='coerce')
            if pd.isna(rec_date):
                continue
            if start is not None and rec_date < start:
                continue
            if end is not None and rec_date > end:
                continue
            if not self._record_matches_focus(rec):
                continue
            filtered.append(rec)

        # Build table rows and filtered totals
        rows = []
        total_dep = 0.0
        total_wit = 0.0
        for r in filtered:
            dep = float(r.get('Deposit', 0.0) or 0.0)
            wit = float(r.get('Withdrawal', 0.0) or 0.0)
            amt = float(r.get('Amount', 0.0) or 0.0)
            total_dep += dep
            total_wit += wit
            rows.append([
                r.get('Date', ''),
                r.get('Description', ''),
                r.get('PaymentAccountDisplay', r.get('Merchant', '')),
                r.get('Category', ''),
                r.get('PaymentMode', ''),
                r.get('SpendBand', ''),
                f"₹{wit:,.2f}",
                f"₹{dep:,.2f}",
                f"₹{amt:,.2f}",
                r.get('Anomaly', 'Normal')
            ])

        self.table_data = rows
        self.filtered_total_deposit = f"₹{total_dep:,.2f}"
        self.filtered_total_withdrawal = f"₹{total_wit:,.2f}"
        self.filtered_net = f"₹{(total_dep - total_wit):,.2f}"

        # Page totals
        try:
            ps = max(1, int(self.page_size))
        except Exception:
            ps = 25
        try:
            pn = max(1, int(self.page_number))
        except Exception:
            pn = 1
        start_idx = (pn - 1) * ps
        end_idx = start_idx + ps
        page_rows = filtered[start_idx:end_idx]
        page_dep = sum(float(r.get('Deposit', 0.0) or 0.0) for r in page_rows)
        page_wit = sum(float(r.get('Withdrawal', 0.0) or 0.0) for r in page_rows)
        self.page_total_deposit = f"₹{page_dep:,.2f}"
        self.page_total_withdrawal = f"₹{page_wit:,.2f}"
        self.page_net = f"₹{(page_dep - page_wit):,.2f}"

    def clear_filter(self):
        self.filter_start = ""
        self.filter_end = ""
        self.page_size = 25
        self.page_number = 1
        self.spend_focus = UNDER_LIMIT_FOCUS
        self.top_n_categories = 6
        self.forecast_model = "exp_smoothing"
        self.exp_smoothing_trend = "add"
        self.exp_smoothing_smoothing_level = 0.2
        self._save_filter_state()
        self.apply_date_filter()

    # -------------------------
    # Main processing pipeline
    # -------------------------
    def process_transactions(self, df: pd.DataFrame, clean_statement_rows: bool = True):
        # normalize columns
        df.columns = df.columns.str.strip()

        # load persisted filter state
        self._load_filter_state()

        # require Date column
        if 'Date' not in df.columns:
            print("CSV must contain a 'Date' column.")
            self._clear_analysis_results()
            return

        # drop Value Dt and similar
        for col in list(df.columns):
            if col.lower() in ('value dt', 'value date', 'value_dt', 'value_date'):
                df.drop(columns=[col], inplace=True, errors='ignore')

        # parse Date (dayfirst common in bank statements)
        df['Date'] = self._parse_statement_dates(df['Date'])

        # description
        if 'Narration' in df.columns:
            df['Description'] = df['Narration'].astype(str)
        elif 'Description' in df.columns:
            df['Description'] = df['Description'].astype(str)
        else:
            df['Description'] = df.iloc[:, 1].astype(str) if df.shape[1] > 1 else "Unknown Transaction"

        # detect withdrawal/deposit columns
        withdrawal_candidates = ['Withdrawal Amt.', 'Withdrawal Amount', 'Debit', 'Dr', 'Debits', 'Withdrawal']
        deposit_candidates = ['Deposit Amt.', 'Deposit Amount', 'Credit', 'Cr', 'Credits', 'Deposit']

        wcol = self._find_column(df, withdrawal_candidates)
        dcol = self._find_column(df, deposit_candidates)

        def to_numeric_col(series):
            return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce').fillna(0.0)

        df['Withdrawal'] = to_numeric_col(df[wcol]) if wcol else 0.0
        df['Deposit'] = to_numeric_col(df[dcol]) if dcol else 0.0

        # amount
        df['Amount'] = df['Deposit'] - df['Withdrawal']

        if clean_statement_rows:
            # Remove statement summary rows robustly.
            summary_keywords = [
                'statement summary', 'opening balance', 'closing bal', 'closing balance', 'debits', 'credits',
                'dr count', 'cr count', 'statement summary :-', 'statement summary -', 'statement summary:'
            ]
            desc_lower = df['Description'].fillna('').str.lower()
            mask_summary_desc = desc_lower.apply(lambda s: any(k in s for k in summary_keywords))
            mask_short_desc = df['Description'].fillna('').str.strip().str.len() <= 3
            med_dep = df['Deposit'].replace(0, np.nan).median(skipna=True)
            med_wit = df['Withdrawal'].replace(0, np.nan).median(skipna=True)
            med_dep = med_dep if not np.isnan(med_dep) else 1.0
            med_wit = med_wit if not np.isnan(med_wit) else 1.0
            factor = 100.0
            mask_large_numbers = ((df['Deposit'].abs() > med_dep * factor) | (df['Withdrawal'].abs() > med_wit * factor))
            mask_summary = mask_summary_desc | (mask_short_desc & mask_large_numbers)
            df = df[~mask_summary].copy()

            # Drop invalid dates, zero movement, and movements outside the analysis limit.
            df = df[df['Date'].notna() & (df['Amount'] != 0)].copy()
            movement_amount = df[['Withdrawal', 'Deposit', 'Amount']].abs().max(axis=1)
            df = df[movement_amount < ANALYSIS_AMOUNT_LIMIT].copy()
        else:
            # These records have already passed the upload-time statement cleanup.
            df = df[df['Date'].notna() & (df['Amount'] != 0)].copy()
        if df.empty:
            self._clear_analysis_results()
            return

        # categories
        df['Category'] = df['Description'].apply(self._assign_category)
        df['PaymentMode'] = df['Description'].apply(self._detect_payment_mode)
        df['SpendBand'] = df['Withdrawal'].apply(self._spend_band)
        df['Merchant'] = df['Description'].apply(self._merchant_name)
        account_parts = [
            self._payment_account_parts(description, merchant)
            for description, merchant in zip(df['Description'], df['Merchant'])
        ]
        df['PaymentAccountKey'] = [part[0] for part in account_parts]
        df['PaymentAccountDisplay'] = [part[1] for part in account_parts]
        repeat_payee_parts = [
            self._repeat_payee_parts(merchant)
            for merchant in df['Merchant']
        ]
        df['RepeatPayeeKey'] = [part[0] for part in repeat_payee_parts]
        df['RepeatPaymentDisplay'] = [part[1] for part in repeat_payee_parts]
        df['RepeatPaymentKey'] = [
            self._repeat_payment_key(payee_key, withdrawal)
            for payee_key, withdrawal in zip(df['RepeatPayeeKey'], df['Withdrawal'])
        ]

        # overall stats
        total_deposits = df.loc[df['Deposit'] > 0, 'Deposit'].sum()
        total_withdrawals = df.loc[df['Withdrawal'] > 0, 'Withdrawal'].sum()
        net_balance = total_deposits - total_withdrawals
        self.total_income = f"₹{total_deposits:,.2f}"
        self.total_expenses = f"₹{total_withdrawals:,.2f}"
        self.net_balance = f"₹{net_balance:,.2f}"

        # monthly aggregation
        df['MonthPeriod'] = df['Date'].dt.to_period('M')
        monthly = df.groupby('MonthPeriod').agg({'Deposit': 'sum', 'Withdrawal': 'sum', 'Amount': 'sum'}).reset_index().sort_values('MonthPeriod')
        monthly['Month'] = monthly['MonthPeriod'].dt.to_timestamp().dt.strftime('%b %Y')
        monthly['Savings'] = monthly['Deposit'] - monthly['Withdrawal']
        self.monthly_chart_data = monthly[['Month', 'Deposit', 'Withdrawal', 'Savings']].rename(columns={'Month': 'month', 'Deposit': 'deposit', 'Withdrawal': 'withdrawal', 'Savings': 'savings'}).to_dict('records')
        self._compute_monthly_essentials(df)
        self._compute_micro_leakage(df, float(total_withdrawals))
        self._compute_repeated_payments(df)

        # category summary & pie
        category_summary = df.groupby('Category').agg({'Withdrawal': 'sum', 'Deposit': 'sum', 'Amount': 'sum'}).reset_index().sort_values('Withdrawal', ascending=False)
        category_summary['WithdrawalDisplay'] = category_summary['Withdrawal'].apply(lambda x: f"₹{x:,.2f}")
        self.category_summary_table = category_summary[['Category', 'WithdrawalDisplay']].values.tolist()
        self.category_chart_data = category_summary[['Category', 'Withdrawal']].rename(columns={'Category': 'name', 'Withdrawal': 'value'}).to_dict('records')

        # stacked category: auto-group small categories into Other and keep top N
        monthly_cat = df.groupby(['MonthPeriod', 'Category']).agg({'Withdrawal': 'sum'}).reset_index()
        if monthly_cat.empty:
            self.monthly_stacked_category_data = []
            self.has_monthly_category_data = False
        else:
            monthly_cat['Month'] = monthly_cat['MonthPeriod'].dt.to_timestamp().dt.strftime('%b %Y')
            totals_by_cat = monthly_cat.groupby('Category')['Withdrawal'].sum().sort_values(ascending=False)
            top_n = max(1, int(self.top_n_categories))
            top_categories = list(totals_by_cat.head(top_n).index)
            monthly_cat['BarCategory'] = monthly_cat['Category'].apply(lambda c: c if c in top_categories else 'Other')
            pivot = monthly_cat.pivot_table(index='Month', columns='BarCategory', values='Withdrawal', aggfunc='sum', fill_value=0)
            pivot = pivot.reset_index().sort_values('Month')
            cols = top_categories + ['Other']
            stacked_records = []
            for _, row in pivot.iterrows():
                rec = {'Month': row['Month']}
                for c in cols:
                    rec[c] = float(row.get(c, 0.0) or 0.0)
                stacked_records.append(rec)
            self.monthly_stacked_category_data = stacked_records
            self.has_monthly_category_data = len(stacked_records) > 0
            self.bar_categories = cols

        # cumulative balance trend
        balance_col = self._find_column(df, ['Closing Balance', 'ClosingBalance', 'Balance', 'Closing Bal'])
        if balance_col and balance_col in df.columns:
            try:
                df[balance_col] = pd.to_numeric(df[balance_col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
                balance_df = df[['Date', balance_col]].dropna().sort_values('Date')
                balance_df = balance_df.groupby('Date').last().reset_index()
                balance_df['date'] = balance_df['Date'].dt.strftime('%Y-%m-%d')
                balance_df = balance_df.rename(columns={balance_col: 'balance'})
                self.chart_data = balance_df[['date', 'balance']].to_dict('records')
            except Exception:
                df_sorted = df.sort_values('Date').copy()
                df_sorted['CumulativeBalance'] = df_sorted['Amount'].cumsum()
                df_sorted['SmoothedBalance'] = df_sorted['CumulativeBalance'].rolling(window=3, min_periods=1).median()
                df_sorted['date'] = df_sorted['Date'].dt.strftime('%Y-%m-%d')
                self.chart_data = df_sorted[['date', 'SmoothedBalance']].rename(columns={'SmoothedBalance': 'balance'}).to_dict('records')
        else:
            df_sorted = df.sort_values('Date').copy()
            df_sorted['CumulativeBalance'] = df_sorted['Amount'].cumsum()
            df_sorted['SmoothedBalance'] = df_sorted['CumulativeBalance'].rolling(window=3, min_periods=1).median()
            df_sorted['date'] = df_sorted['Date'].dt.strftime('%Y-%m-%d')
            self.chart_data = df_sorted[['date', 'SmoothedBalance']].rename(columns={'SmoothedBalance': 'balance'}).to_dict('records')

        # anomaly detection
        df['Anomaly'] = 'Normal'
        if len(df) > 5:
            X = df['Amount'].values.reshape(-1, 1)
            model = IsolationForest(contamination=0.07, random_state=42)
            preds = model.fit_predict(X)
            df.loc[preds == -1, 'Anomaly'] = '⚠️ Outlier'

        # top expenses
        top_expenses = df[df['Withdrawal'] > 0].sort_values('Withdrawal', ascending=False).head(10)
        top5_display = top_expenses[['Date', 'Description', 'Category', 'PaymentMode', 'SpendBand', 'Withdrawal']].copy()
        top5_display['Date'] = top5_display['Date'].dt.strftime('%Y-%m-%d')
        top5_display['Withdrawal'] = top5_display['Withdrawal'].apply(lambda x: f"₹{x:,.2f}")
        self.top_expenses_table = top5_display.values.tolist()

        # recurring payments detection
        desc_amount = df[df['Withdrawal'] > 0][['Merchant', 'Description', 'Withdrawal']].copy()
        desc_amount['desc_norm'] = desc_amount['Merchant'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str[:60]
        recurring = []
        recurring_total_value = 0.0
        grouped = desc_amount.groupby('desc_norm')
        for name, group in grouped:
            if len(group) >= 3:
                amounts = group['Withdrawal'].round(2).tolist()
                total_for_desc = float(group['Withdrawal'].sum())
                recurring_total_value += total_for_desc
                sample_amounts = ','.join(map(lambda v: f"₹{v:,.2f}", amounts[:3]))
                recurring.append([name, str(len(group)), sample_amounts, f"₹{total_for_desc:,.2f}"])
        self.recurring_payments = recurring
        self.recurring_total = f"₹{recurring_total_value:,.2f}"

        # transaction table and raw_transactions
        formatted = df.sort_values('Date').copy()
        formatted['Date'] = formatted['Date'].dt.strftime('%Y-%m-%d')
        raw_records = []
        rows = []
        total_dep = 0.0
        total_wit = 0.0
        for _, row in formatted.iterrows():
            rec = {
                'Date': row['Date'],
                'Description': row.get('Description', ''),
                'Category': row.get('Category', ''),
                'PaymentMode': row.get('PaymentMode', ''),
                'SpendBand': row.get('SpendBand', ''),
                'Merchant': row.get('Merchant', ''),
                'PaymentAccountKey': row.get('PaymentAccountKey', ''),
                'PaymentAccountDisplay': row.get('PaymentAccountDisplay', ''),
                'RepeatPayeeKey': row.get('RepeatPayeeKey', ''),
                'RepeatPaymentDisplay': row.get('RepeatPaymentDisplay', ''),
                'RepeatPaymentKey': row.get('RepeatPaymentKey', ''),
                'Withdrawal': float(row.get('Withdrawal') or 0.0),
                'Deposit': float(row.get('Deposit') or 0.0),
                'Amount': float(row.get('Amount') or 0.0),
                'Anomaly': row.get('Anomaly', 'Normal')
            }
            raw_records.append(rec)
            total_dep += rec['Deposit']
            total_wit += rec['Withdrawal']
            rows.append([
                rec['Date'],
                rec['Description'],
                rec['PaymentAccountDisplay'],
                rec['Category'],
                rec['PaymentMode'],
                rec['SpendBand'],
                f"₹{rec['Withdrawal']:,.2f}",
                f"₹{rec['Deposit']:,.2f}",
                f"₹{rec['Amount']:,.2f}",
                rec['Anomaly'],
            ])
        self.raw_transactions = raw_records
        self.table_data = rows
        self.filtered_total_deposit = f"₹{total_dep:,.2f}"
        self.filtered_total_withdrawal = f"₹{total_wit:,.2f}"
        self.filtered_net = f"₹{(total_dep - total_wit):,.2f}"

        # page totals default
        try:
            ps = max(1, int(self.page_size))
        except Exception:
            ps = 25
        try:
            pn = max(1, int(self.page_number))
        except Exception:
            pn = 1
        start_idx = (pn - 1) * ps
        end_idx = start_idx + ps
        page_rows = raw_records[start_idx:end_idx]
        page_dep = sum(r['Deposit'] for r in page_rows)
        page_wit = sum(r['Withdrawal'] for r in page_rows)
        self.page_total_deposit = f"₹{page_dep:,.2f}"
        self.page_total_withdrawal = f"₹{page_wit:,.2f}"
        self.page_net = f"₹{(page_dep - page_wit):,.2f}"

        # monthly summary table
        monthly_display = monthly.copy()
        monthly_display['Deposit'] = monthly_display['Deposit'].apply(lambda x: f"₹{x:,.2f}")
        monthly_display['Withdrawal'] = monthly_display['Withdrawal'].apply(lambda x: f"₹{x:,.2f}")
        monthly_display['Savings'] = monthly_display['Savings'].apply(lambda x: f"₹{x:,.2f}")
        self.monthly_summary_table = monthly_display[['Month', 'Deposit', 'Withdrawal', 'Savings']].values.tolist()

        # forecast
        self._compute_forecast(monthly)
        self.apply_date_filter()

    # -------------------------
    # Forecast helpers
    # -------------------------
    def _compute_forecast(self, monthly_df: pd.DataFrame):
        try:
            if len(monthly_df) < 2:
                self.forecast_table = [["Next Month (forecast)", "N/A", "N/A", "N/A"]]
                return
            monthly_numeric = monthly_df.reset_index(drop=True)
            if self.forecast_model == "linear":
                monthly_numeric['idx'] = np.arange(len(monthly_numeric))
                X = monthly_numeric[['idx']].values
                lr_dep = LinearRegression().fit(X, monthly_numeric['Deposit'].values)
                lr_wit = LinearRegression().fit(X, monthly_numeric['Withdrawal'].values)
                next_idx = np.array([[len(monthly_numeric)]])
                pred_dep = max(0.0, lr_dep.predict(next_idx)[0])
                pred_wit = max(0.0, lr_wit.predict(next_idx)[0])
            else:
                try:
                    dep_series = monthly_numeric['Deposit'].astype(float).values
                    trend = None if self.exp_smoothing_trend in ("None", "none", "") else self.exp_smoothing_trend
                    model_dep = ExponentialSmoothing(dep_series, trend=trend, seasonal=None, initialization_method="estimated")
                    fit_dep = model_dep.fit(smoothing_level=self.exp_smoothing_smoothing_level, optimized=True)
                    pred_dep = max(0.0, float(fit_dep.forecast(1)[0]))
                except Exception:
                    pred_dep = float(monthly_numeric['Deposit'].iloc[-1])
                try:
                    wit_series = monthly_numeric['Withdrawal'].astype(float).values
                    trend = None if self.exp_smoothing_trend in ("None", "none", "") else self.exp_smoothing_trend
                    model_wit = ExponentialSmoothing(wit_series, trend=trend, seasonal=None, initialization_method="estimated")
                    fit_wit = model_wit.fit(smoothing_level=self.exp_smoothing_smoothing_level, optimized=True)
                    pred_wit = max(0.0, float(fit_wit.forecast(1)[0]))
                except Exception:
                    pred_wit = float(monthly_numeric['Withdrawal'].iloc[-1])
            pred_sav = pred_dep - pred_wit
            self.forecast_table = [["Next Month (forecast)", f"₹{pred_dep:,.2f}", f"₹{pred_wit:,.2f}", f"₹{pred_sav:,.2f}"]]
        except Exception:
            self.forecast_table = [["Next Month (forecast)", "N/A", "N/A", "N/A"]]

# -------------------------
# UI Components (single-column full-width layout)
# -------------------------
def navbar() -> rx.Component:
    return rx.hstack(
        rx.vstack(
            rx.heading("Monthly Spend Analyzer", size="6", weight="bold"),
            rx.text("Monthly spend, repeated payees, category charts, and transaction drilldowns", size="2", color="gray"),
            align="start",
            spacing="1",
        ),
        rx.color_mode.button(),
        justify="between",
        align="center",
        padding="1.25em 1.5em",
        border_bottom="1px solid var(--gray-a4)",
        width="100%",
        background_color="var(--gray-1)",
    )

def stats_card(title: str, value: str, color: str) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.text(title, size="2", color="gray"),
            rx.heading(value, size="7", color=color),
            align="start",
            spacing="2",
        ),
        width="100%",
        min_height="118px",
        border="1px solid var(--gray-a4)",
    )

def dashboard_stats() -> rx.Component:
    return rx.hstack(
        stats_card(AnalyzerState.latest_month_label, AnalyzerState.latest_month_total, "blue"),
        stats_card("Below ₹5,000", AnalyzerState.latest_month_under_5k, "orange"),
        stats_card("₹5,000 to ₹10,000", AnalyzerState.latest_month_5k_to_10k, "purple"),
        stats_card("Recurring This Month", AnalyzerState.latest_month_recurring, "red"),
        spacing="4",
        width="100%",
        wrap="wrap",
    )

def upload_section() -> rx.Component:
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=32, color="gray"),
                rx.text("Drag & Drop CSV or Excel Statement", size="4"),
                rx.text("This dashboard groups repeated payee amounts, categories, monthly trends, and transaction details.", size="2", color="gray"),
                align="center",
                justify="center",
                padding="3em",
                border="2px dashed var(--gray-a6)",
                border_radius="md",
                width="100%",
                background_color="var(--gray-2)",
            ),
            id="statement_upload",
            accept={
                "text/csv": [".csv"],
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
                "application/vnd.ms-excel": [".xls"],
            },
            max_files=1,
            width="100%",
        ),
        rx.hstack(
            rx.button("Analyze Statement", on_click=AnalyzerState.handle_upload(rx.upload_files(upload_id="statement_upload")), size="3"),
            rx.cond(AnalyzerState.is_processing, rx.spinner(size="3")),
            wrap="wrap",
        ),
        align="center",
        width="100%",
        padding_y="2em",
    )

def micro_metric_card(title: str, value: str, helper: str, color: str) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.text(title, size="2", color="gray"),
            rx.heading(value, size="6", color=color),
            rx.text(helper, size="2", color="gray"),
            align="start",
            spacing="2",
        ),
        min_width="190px",
        flex="1",
        border_left=f"4px solid {color}",
        border="1px solid var(--gray-a4)",
    )

def micro_leakage_report() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.heading("Micro Money Leakage", size="5"),
                rx.text("Sub-₹5,000 withdrawals, UPI spends, and food/restaurant repeats", size="2", color="gray"),
                align="start",
                spacing="1",
            ),
            rx.select(
                items=SPEND_FOCUS_OPTIONS,
                value=AnalyzerState.spend_focus,
                on_change=AnalyzerState.set_spend_focus,
            ),
            justify="between",
            align="center",
            width="100%",
            wrap="wrap",
        ),
        rx.hstack(
            micro_metric_card("Micro Total", AnalyzerState.micro_expense_total, AnalyzerState.micro_expense_count_label, "#f97316"),
            micro_metric_card("Micro Share", AnalyzerState.micro_expense_share, "of all withdrawals", "#ef4444"),
            micro_metric_card("Average Ticket", AnalyzerState.micro_average_ticket, "below ₹5,000", "#3b82f6"),
            micro_metric_card("UPI Micro", AnalyzerState.upi_micro_total, AnalyzerState.upi_micro_count_label, "#8b5cf6"),
            micro_metric_card("Food/Restaurant Micro", AnalyzerState.restaurant_micro_total, AnalyzerState.restaurant_micro_count_label, "#22c55e"),
            width="100%",
            spacing="4",
            align="stretch",
            wrap="wrap",
        ),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.heading("Amount Bands", size="4"),
                    rx.data_table(
                        data=AnalyzerState.amount_band_summary_table,
                        columns=["Band", "Count", "Total", "Average", "Share", "UPI Total", "Food Total"],
                        pagination=False,
                        width="100%",
                        style={"width": "100%", "tableLayout": "fixed"},
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Micro Categories", size="4"),
                    rx.recharts.pie_chart(
                        rx.recharts.pie(data_key="value", name_key="name", cx="50%", cy="50%", outer_radius=92, label=True),
                        rx.recharts.tooltip(),
                        data=AnalyzerState.micro_category_chart_data,
                        height=300,
                        width="100%",
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            width="100%",
            spacing="4",
            align="stretch",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Monthly Micro Spend", size="4"),
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="MicroSpend", name="Micro Spend", stroke="#f97316", type_="monotone"),
                    rx.recharts.line(data_key="UPI", name="UPI", stroke="#8b5cf6", type_="monotone"),
                    rx.recharts.line(data_key="Food", name="Food", stroke="#22c55e", type_="monotone"),
                    rx.recharts.x_axis(data_key="Month"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.tooltip(),
                    data=AnalyzerState.micro_monthly_chart_data,
                    height=320,
                    width="100%",
                ),
                width="100%",
            ),
            width="100%",
            border="1px solid var(--gray-a4)",
        ),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.heading("Repeat Micro Merchants", size="4"),
                    rx.data_table(
                        data=AnalyzerState.micro_leakage_table,
                        columns=["Merchant", "Category", "Mode", "Count", "Total", "Average", "Largest"],
                        pagination=True,
                        search=True,
                        sort=True,
                        width="100%",
                        style={"width": "100%", "tableLayout": "fixed"},
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Leakage Signals", size="4"),
                    rx.data_table(
                        data=AnalyzerState.micro_insight_table,
                        columns=["Signal", "Value", "Detail"],
                        pagination=False,
                        width="100%",
                        style={"width": "100%", "tableLayout": "fixed"},
                    ),
                    rx.data_table(
                        data=AnalyzerState.micro_category_table,
                        columns=["Category", "Count", "Total", "Average", "Share"],
                        pagination=True,
                        width="100%",
                        style={"width": "100%", "tableLayout": "fixed"},
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            width="100%",
            spacing="4",
            align="stretch",
        ),
        width="100%",
        spacing="4",
        padding_top="1em",
    )

def monthly_spend_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.row_header_cell(row[0]),
        rx.table.cell(
            rx.button(
                row[1],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_month_details(row[0], DETAIL_UNDER_5K),
            )
        ),
        rx.table.cell(
            rx.button(
                row[2],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_month_details(row[0], DETAIL_5K_TO_10K),
            )
        ),
        rx.table.cell(
            rx.button(
                row[3],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_month_details(row[0], DETAIL_ALL_BANDS),
            )
        ),
        rx.table.cell(row[4]),
    )

def recurring_payment_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.row_header_cell(
            rx.button(
                row[0],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_merchant_details(row[0]),
            )
        ),
        rx.table.cell(row[1]),
        rx.table.cell(row[2]),
        rx.table.cell(row[3]),
        rx.table.cell(row[4]),
        rx.table.cell(
            rx.button(
                row[5],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_merchant_details(row[0]),
            )
        ),
        rx.table.cell(row[6]),
        rx.table.cell(row[7]),
    )

def monthly_spend_click_table() -> rx.Component:
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Month"),
                    rx.table.column_header_cell("Below ₹5,000"),
                    rx.table.column_header_cell("₹5,000-₹10,000"),
                    rx.table.column_header_cell("Total"),
                    rx.table.column_header_cell("Payments"),
                )
            ),
            rx.table.body(
                rx.foreach(AnalyzerState.monthly_spend_table, monthly_spend_row)
            ),
            width="100%",
        ),
        width="100%",
        overflow_x="auto",
    )

def recurring_payment_click_table() -> rx.Component:
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Merchant"),
                    rx.table.column_header_cell("Category"),
                    rx.table.column_header_cell("Mode"),
                    rx.table.column_header_cell("Months Paid"),
                    rx.table.column_header_cell("Avg Monthly"),
                    rx.table.column_header_cell("Total Paid"),
                    rx.table.column_header_cell("Monthly Range"),
                    rx.table.column_header_cell("Last Paid"),
                )
            ),
            rx.table.body(
                rx.foreach(AnalyzerState.monthly_recurring_table, recurring_payment_row)
            ),
            width="100%",
        ),
        width="100%",
        overflow_x="auto",
    )

def repeated_payment_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.row_header_cell(
            rx.button(
                row[1],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_repeated_payment_details(row[0]),
            )
        ),
        rx.table.cell(
            rx.button(
                row[2],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_repeated_payment_details(row[0]),
            )
        ),
        rx.table.cell(row[3]),
        rx.table.cell(row[4]),
        rx.table.cell(row[5]),
        rx.table.cell(row[6]),
        rx.table.cell(row[7]),
        rx.table.cell(row[8]),
        rx.table.cell(row[9]),
    )

def repeated_payment_click_table() -> rx.Component:
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Account/Payee"),
                    rx.table.column_header_cell("Amount"),
                    rx.table.column_header_cell("Times"),
                    rx.table.column_header_cell("Total"),
                    rx.table.column_header_cell("Category"),
                    rx.table.column_header_cell("Mode"),
                    rx.table.column_header_cell("Months"),
                    rx.table.column_header_cell("First Paid"),
                    rx.table.column_header_cell("Last Paid"),
                )
            ),
            rx.table.body(
                rx.foreach(AnalyzerState.repeated_payment_table, repeated_payment_row)
            ),
            width="100%",
        ),
        width="100%",
        overflow_x="auto",
    )

def detail_recurring_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.row_header_cell(
            rx.button(
                row[0],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_merchant_details(row[0]),
            )
        ),
        rx.table.cell(row[1]),
        rx.table.cell(row[2]),
        rx.table.cell(
            rx.button(
                row[3],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_merchant_details(row[0]),
            )
        ),
        rx.table.cell(row[4]),
        rx.table.cell(row[5]),
        rx.table.cell(row[6]),
        rx.table.cell(row[7]),
    )

def detail_transaction_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.cell(row[0]),
        rx.table.row_header_cell(
            rx.button(
                row[1],
                variant="ghost",
                size="2",
                on_click=AnalyzerState.show_merchant_details(row[2]),
            )
        ),
        rx.table.cell(row[2]),
        rx.table.cell(row[3]),
        rx.table.cell(row[4]),
        rx.table.cell(row[5]),
        rx.table.cell(row[6]),
        rx.table.cell(row[7]),
    )

def detail_recurring_click_table() -> rx.Component:
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Merchant"),
                    rx.table.column_header_cell("Category"),
                    rx.table.column_header_cell("Mode"),
                    rx.table.column_header_cell("This Month"),
                    rx.table.column_header_cell("Payments"),
                    rx.table.column_header_cell("Months Paid"),
                    rx.table.column_header_cell("All Months Total"),
                    rx.table.column_header_cell("Last Date"),
                )
            ),
            rx.table.body(
                rx.foreach(AnalyzerState.detail_recurring_table, detail_recurring_row)
            ),
            width="100%",
        ),
        width="100%",
        overflow_x="auto",
    )

def detail_transaction_click_table() -> rx.Component:
    return rx.box(
        rx.table.root(
            rx.table.header(
                rx.table.row(
                    rx.table.column_header_cell("Date"),
                    rx.table.column_header_cell("Account/Payee"),
                    rx.table.column_header_cell("Merchant"),
                    rx.table.column_header_cell("Description"),
                    rx.table.column_header_cell("Category"),
                    rx.table.column_header_cell("Mode"),
                    rx.table.column_header_cell("Band"),
                    rx.table.column_header_cell("Amount"),
                )
            ),
            rx.table.body(
                rx.foreach(AnalyzerState.detail_table, detail_transaction_row)
            ),
            width="100%",
        ),
        width="100%",
        overflow_x="auto",
    )

def transaction_detail_panel() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading(AnalyzerState.detail_title, size="5"),
                    rx.text(AnalyzerState.detail_summary, size="2", color="gray"),
                    align="start",
                    spacing="1",
                ),
                rx.hstack(
                    rx.vstack(
                        rx.text(AnalyzerState.detail_count_label, size="2", color="gray"),
                        rx.heading(AnalyzerState.detail_total, size="5", color="blue"),
                        align="end",
                        spacing="1",
                    ),
                    rx.select(
                        items=DETAIL_BAND_FILTER_OPTIONS,
                        value=AnalyzerState.detail_band_filter,
                        on_change=AnalyzerState.set_detail_band_filter,
                    ),
                    align="center",
                    spacing="4",
                    wrap="wrap",
                ),
                justify="between",
                align="center",
                width="100%",
                wrap="wrap",
            ),
            rx.cond(
                AnalyzerState.detail_scope == "month",
                rx.card(
                    rx.vstack(
                        rx.heading("Recurring Payments In This Month", size="4"),
                        rx.text("Merged by merchant. Click a merchant to see every payment date and amount.", size="2", color="gray"),
                        detail_recurring_click_table(),
                        width="100%",
                        spacing="3",
                    ),
                    width="100%",
                    border="1px solid var(--gray-a4)",
                ),
                rx.box(),
            ),
            rx.vstack(
                rx.heading("Payment List", size="4"),
                rx.text("Click a merchant in the list to open that individual payment history.", size="2", color="gray"),
                detail_transaction_click_table(),
                width="100%",
                spacing="3",
            ),
            width="100%",
            spacing="4",
        ),
        width="100%",
        border="1px solid var(--gray-a4)",
    )

def monthly_essentials_report() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading("Monthly Spending Below ₹10,000", size="5"),
                    rx.text("Split into payments below ₹5,000 and ₹5,000 to below ₹10,000", size="2", color="gray"),
                    align="start",
                    spacing="1",
                ),
                rx.badge("₹10,000 and above hidden", color_scheme="blue", variant="soft"),
                justify="between",
                align="center",
                width="100%",
                wrap="wrap",
            ),
            rx.hstack(
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="Under5k", name="Below ₹5,000", fill="#f97316", stack_id="spend"),
                    rx.recharts.bar(data_key="FiveToTenK", name="₹5,000-₹10,000", fill="#3b82f6", stack_id="spend"),
                    rx.recharts.x_axis(data_key="Month"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.tooltip(),
                    data=AnalyzerState.monthly_spend_chart_data,
                    height=340,
                    width="100%",
                ),
                monthly_spend_click_table(),
                width="100%",
                spacing="4",
                align="stretch",
                wrap="wrap",
            ),
            width="100%",
            spacing="4",
        ),
        width="100%",
        border="1px solid var(--gray-a4)",
    )

def repeated_payment_report() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.heading("Same Payee and Amount Repeats", size="5"),
                rx.text("Repeated withdrawals grouped by the first two payee words and amount", size="2", color="gray"),
                align="start",
                spacing="1",
            ),
            rx.vstack(
                rx.text(AnalyzerState.repeated_payment_count_label, size="2", color="gray"),
                rx.heading(AnalyzerState.repeated_payment_total, size="5", color="red"),
                align="end",
                spacing="1",
            ),
            justify="between",
            align="center",
            width="100%",
            wrap="wrap",
        ),
        rx.hstack(
            micro_metric_card("Repeat Groups", AnalyzerState.repeated_payment_count_label, AnalyzerState.repeated_payment_top_label, "#0ea5e9"),
            micro_metric_card("Repeat Total", AnalyzerState.repeated_payment_total, "same first-two-word payee and amount", "#ef4444"),
            width="100%",
            spacing="4",
            align="stretch",
            wrap="wrap",
        ),
        rx.hstack(
            rx.card(
                rx.vstack(
                    rx.heading("Repeated Payments by Category", size="4"),
                    rx.recharts.bar_chart(
                        rx.recharts.bar(data_key="Total", name="Repeated Total", fill="#0ea5e9"),
                        rx.recharts.x_axis(data_key="Category"),
                        rx.recharts.y_axis(),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                        rx.recharts.tooltip(),
                        data=AnalyzerState.repeated_payment_category_chart_data,
                        height=320,
                        width="100%",
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            rx.card(
                rx.vstack(
                    rx.heading("Repeated Payments by Mode", size="4"),
                    rx.recharts.pie_chart(
                        rx.recharts.pie(data_key="value", name_key="name", cx="50%", cy="50%", outer_radius=92, label=True),
                        rx.recharts.tooltip(),
                        data=AnalyzerState.repeated_payment_mode_chart_data,
                        height=320,
                        width="100%",
                    ),
                    width="100%",
                ),
                width="100%",
                border="1px solid var(--gray-a4)",
            ),
            width="100%",
            spacing="4",
            align="stretch",
            wrap="wrap",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Monthly Payee Repeats", size="4"),
                rx.recharts.bar_chart(
                    rx.recharts.bar(data_key="RepeatedSpend", name="Repeated Spend", fill="#14b8a6"),
                    rx.recharts.bar(data_key="Payments", name="Payments", fill="#f97316"),
                    rx.recharts.x_axis(data_key="Month"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.tooltip(),
                    data=AnalyzerState.repeated_payment_monthly_chart_data,
                    height=300,
                    width="100%",
                ),
                width="100%",
            ),
            width="100%",
            border="1px solid var(--gray-a4)",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Payee Repeat Table", size="4"),
                repeated_payment_click_table(),
                width="100%",
                spacing="3",
            ),
            width="100%",
            border="1px solid var(--gray-a4)",
        ),
        width="100%",
        spacing="4",
    )

def recurring_monthly_report() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading("Repeat Monthly Payments", size="5"),
                    rx.text("Merchants paid in two or more months, based on the cleaned merchant name", size="2", color="gray"),
                    align="start",
                    spacing="1",
                ),
                rx.vstack(
                    rx.text(AnalyzerState.recurring_monthly_count_label, size="2", color="gray"),
                    rx.heading(AnalyzerState.recurring_monthly_total, size="5", color="red"),
                    align="end",
                    spacing="1",
                ),
                justify="between",
                align="center",
                width="100%",
                wrap="wrap",
            ),
            rx.recharts.bar_chart(
                rx.recharts.bar(data_key="RecurringSpend", name="Recurring Spend", fill="#ef4444"),
                rx.recharts.x_axis(data_key="Month"),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                rx.recharts.tooltip(),
                data=AnalyzerState.monthly_recurring_chart_data,
                height=300,
                width="100%",
            ),
            recurring_payment_click_table(),
            width="100%",
            spacing="4",
        ),
        width="100%",
        border="1px solid var(--gray-a4)",
    )

def monthly_bar_chart() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Monthly Deposits vs Withdrawals vs Savings", size="4"),
            rx.recharts.bar_chart(
                rx.recharts.bar(data_key="deposit", name="Deposit", fill="#2ecc71"),
                rx.recharts.bar(data_key="withdrawal", name="Withdrawal", fill="#e74c3c"),
                rx.recharts.bar(data_key="savings", name="Savings", fill="#3498db"),
                rx.recharts.x_axis(data_key="month"),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                rx.recharts.tooltip(),
                data=AnalyzerState.monthly_chart_data,
                height=360,
                width="100%",
            ),
            width="100%",
        ),
        width="100%",
        margin_y="2em",
    )

def monthly_stacked_category_chart() -> rx.Component:
    bars = [
        rx.recharts.bar(data_key=cat, stack_id="a", name=cat, fill=CATEGORY_COLORS.get(cat, "#64748b"))
        for cat in DEFAULT_BAR_CATEGORIES
    ]
    bars.append(rx.recharts.bar(data_key="Other", stack_id="a", name="Other", fill=CATEGORY_COLORS["Other"]))
    return rx.cond(
        AnalyzerState.has_monthly_category_data,
        rx.card(
            rx.vstack(
                rx.heading("Monthly Withdrawals by Category (stacked)", size="4"),
                rx.recharts.bar_chart(
                    *bars,
                    rx.recharts.x_axis(data_key="Month"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                    rx.recharts.tooltip(),
                    data=AnalyzerState.monthly_stacked_category_data,
                    height=420,
                    width="100%",
                ),
                width="100%",
            ),
            width="100%",
            margin_y="2em",
        ),
        rx.card(
            rx.vstack(
                rx.heading("Monthly Withdrawals by Category (stacked)", size="4"),
                rx.text("No category-by-month data available.", size="2", color="gray"),
            ),
            width="100%",
            margin_y="2em",
        ),
    )

def trend_chart() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Cumulative Balance Trend", size="4"),
            rx.recharts.line_chart(
                rx.recharts.line(data_key="balance", stroke="#6c5ce7", type_="monotone"),
                rx.recharts.x_axis(data_key="date"),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                rx.recharts.tooltip(),
                data=AnalyzerState.chart_data,
                height=320,
                width="100%",
            ),
            width="100%",
        ),
        width="100%",
        margin_y="2em",
    )

def monthly_summary_card() -> rx.Component:
    # Ensure table uses full width and fixed layout so columns stretch
    return rx.card(
        rx.vstack(
            rx.heading("Monthly Summary", size="4"),
            rx.data_table(
                data=AnalyzerState.monthly_summary_table,
                columns=["Month", "Deposit", "Withdrawal", "Savings"],
                pagination=True,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
        ),
        width="100%",
    )

def top_expenses_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Top Expenses", size="4"),
            rx.data_table(
                data=AnalyzerState.top_expenses_table,
                columns=["Date", "Description", "Category", "Mode", "Band", "Amount"],
                pagination=True,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
        ),
        width="100%",
    )

def recurring_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Detected Recurring Payments", size="4"),
            rx.hstack(rx.text("Recurring Total:", size="2", color="gray"), rx.heading(AnalyzerState.recurring_total, size="4")),
            rx.data_table(
                data=AnalyzerState.recurring_payments,
                columns=["Description Sample", "Count", "Amounts Sample", "Total for Description"],
                pagination=True,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
        ),
        width="100%",
    )

def forecast_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Forecast (next month)", size="4"),
            rx.hstack(
                rx.text("Model", size="2", color="gray"),
                rx.select(items=["exp_smoothing", "linear"], value=AnalyzerState.forecast_model, on_change=AnalyzerState.set_forecast_model),
                rx.text("Trend", size="2", color="gray"),
                rx.select(items=["add", "mul", "None"], value=AnalyzerState.exp_smoothing_trend, on_change=AnalyzerState.set_exp_smoothing_trend),
                rx.text("Alpha", size="2", color="gray"),
                rx.input(type="number", value=AnalyzerState.exp_smoothing_smoothing_level, on_change=AnalyzerState.set_exp_smoothing_smoothing_level),
                rx.button("Recompute Forecast", on_click=AnalyzerState.apply_date_filter, size="3"),
            ),
            rx.data_table(
                data=AnalyzerState.forecast_table,
                columns=["Period", "Deposit (forecast)", "Withdrawal (forecast)", "Savings (forecast)"],
                pagination=False,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
        ),
        width="100%",
    )

def category_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Category Breakdown (Withdrawals)", size="4"),
            rx.recharts.pie_chart(
                rx.recharts.pie(data_key="value", name_key="name", cx="50%", cy="50%", outer_radius=80, label=True),
                rx.recharts.tooltip(),
                data=AnalyzerState.category_chart_data,
                height=320,
                width="100%",
            ),
            rx.data_table(
                data=AnalyzerState.category_summary_table,
                columns=["Category", "Total Withdrawals"],
                pagination=True,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
            rx.hstack(rx.text("Top N categories for stacked chart", size="2", color="gray"), rx.input(type="number", value=AnalyzerState.top_n_categories, on_change=AnalyzerState.set_top_n_categories)),
            rx.upload(
                rx.vstack(rx.text("Upload category rules CSV (keyword,category)"), rx.text("Example: netflix,Bills", size="2", color="gray")),
                id="category_rules_upload",
                accept={"text/csv": [".csv"]},
                max_files=1,
                width="100%",
            ),
            rx.hstack(
                rx.button("Load Category Rules", on_click=AnalyzerState.handle_category_rules_upload(rx.upload_files(upload_id="category_rules_upload")), size="3"),
                rx.text("Upload a CSV with rows: keyword,category", size="2", color="gray"),
            ),
        ),
        width="100%",
    )

def transaction_filter_controls() -> rx.Component:
    return rx.vstack(
        rx.heading("Transaction Filters", size="4"),
        rx.text("Amounts ₹10,000 and above are hidden from analysis.", size="2", color="gray"),
        rx.hstack(
            rx.vstack(rx.text("Start Date", size="2", color="gray"), rx.input(type="date", value=AnalyzerState.filter_start, on_change=AnalyzerState.set_filter_start), align="start"),
            rx.vstack(rx.text("End Date", size="2", color="gray"), rx.input(type="date", value=AnalyzerState.filter_end, on_change=AnalyzerState.set_filter_end), align="start"),
            rx.vstack(rx.text("Spend Focus", size="2", color="gray"), rx.select(items=SPEND_FOCUS_OPTIONS, value=AnalyzerState.spend_focus, on_change=AnalyzerState.set_spend_focus), align="start"),
            rx.vstack(rx.text("Page Size", size="2", color="gray"), rx.input(type="number", value=AnalyzerState.page_size, on_change=AnalyzerState.set_page_size), align="start"),
            rx.vstack(rx.text("Page Number", size="2", color="gray"), rx.input(type="number", value=AnalyzerState.page_number, on_change=AnalyzerState.set_page_number), align="start"),
            rx.vstack(rx.button("Apply Filter", on_click=AnalyzerState.apply_date_filter, size="3")),
            rx.vstack(rx.button("Clear Filter", on_click=AnalyzerState.clear_filter, size="3")),
            spacing="4",
            align="end",
            wrap="wrap",
        ),
        width="100%",
    )

def transaction_totals_row() -> rx.Component:
    return rx.hstack(
        rx.vstack(rx.text("Filtered Totals", size="2", color="gray"), rx.text(" ")),
        rx.vstack(rx.text("Deposits", size="2", color="gray"), rx.heading(AnalyzerState.filtered_total_deposit, size="4", color="green")),
        rx.vstack(rx.text("Withdrawals", size="2", color="gray"), rx.heading(AnalyzerState.filtered_total_withdrawal, size="4", color="red")),
        rx.vstack(rx.text("Net", size="2", color="gray"), rx.heading(AnalyzerState.filtered_net, size="4", color="blue")),
        spacing="8",
        width="100%",
        justify="between",
    )

def transaction_page_totals_row() -> rx.Component:
    return rx.hstack(
        rx.vstack(rx.text("Page Totals (visible page)", size="2", color="gray"), rx.text(" ")),
        rx.vstack(rx.text("Deposits", size="2", color="gray"), rx.heading(AnalyzerState.page_total_deposit, size="4", color="green")),
        rx.vstack(rx.text("Withdrawals", size="2", color="gray"), rx.heading(AnalyzerState.page_total_withdrawal, size="4", color="red")),
        rx.vstack(rx.text("Net", size="2", color="gray"), rx.heading(AnalyzerState.page_net, size="4", color="blue")),
        spacing="8",
        width="100%",
        justify="between",
    )

def transaction_table_card() -> rx.Component:
    # Full-width table with fixed layout so columns expand to fill screen
    return rx.card(
        rx.vstack(
            rx.heading("Transaction History", size="4"),
            transaction_filter_controls(),
            transaction_totals_row(),
            transaction_page_totals_row(),
            rx.data_table(
                data=AnalyzerState.table_data,
                columns=AnalyzerState.table_columns,
                pagination=True,
                search=True,
                sort=True,
                width="100%",
                style={"width": "100%", "tableLayout": "fixed"},
            ),
        ),
        width="100%",
    )

def chat_message_bubble(row) -> rx.Component:
    return rx.cond(
        row[0] == "user",
        rx.hstack(
            rx.box(
                rx.text(row[1], size="2", white_space="pre-line"),
                max_width="86%",
                padding="0.75em 0.9em",
                border_radius="8px",
                background_color="#0f766e",
                color="white",
            ),
            justify="end",
            width="100%",
        ),
        rx.hstack(
            rx.box(
                rx.text(row[1], size="2", white_space="pre-line"),
                max_width="86%",
                padding="0.75em 0.9em",
                border_radius="8px",
                background_color="var(--gray-3)",
                border="1px solid var(--gray-a4)",
            ),
            justify="start",
            width="100%",
        ),
    )

def payment_chat_widget() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.cond(
                AnalyzerState.chat_is_open,
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.vstack(
                                rx.heading("Payment Chat", size="4"),
                                rx.text("Ask from the loaded statement", size="2", color="gray"),
                                align="start",
                                spacing="1",
                            ),
                            rx.button(
                                rx.icon("x", size=16),
                                on_click=AnalyzerState.close_chat,
                                variant="ghost",
                                size="2",
                            ),
                            justify="between",
                            align="center",
                            width="100%",
                        ),
                        rx.box(
                            rx.vstack(
                                rx.foreach(AnalyzerState.chat_messages, chat_message_bubble),
                                spacing="3",
                                width="100%",
                            ),
                            id="payment-chat-scroll",
                            height="310px",
                            overflow_y="auto",
                            width="100%",
                            padding_right="0.2em",
                        ),
                        rx.form(
                            rx.hstack(
                                rx.input(
                                    name="chat_question",
                                    placeholder="Ask about payments, deposits, merchants",
                                    value=AnalyzerState.chat_input,
                                    on_change=AnalyzerState.set_chat_input,
                                    width="100%",
                                ),
                                rx.button(
                                    rx.icon("send", size=16),
                                    "Ask",
                                    type="submit",
                                    size="2",
                                ),
                                width="100%",
                                spacing="2",
                            ),
                            on_submit=AnalyzerState.ask_chat_form,
                            reset_on_submit=False,
                            width="100%",
                        ),
                        width="100%",
                        spacing="3",
                    ),
                    width="min(390px, calc(100vw - 2rem))",
                    border="1px solid var(--gray-a5)",
                    box_shadow="0 18px 50px rgba(15, 23, 42, 0.22)",
                    background_color="var(--gray-1)",
                ),
                rx.box(),
            ),
            rx.hstack(
                rx.button(
                    rx.icon("message-circle", size=18),
                    "Chat",
                    on_click=AnalyzerState.toggle_chat,
                    size="3",
                    border_radius="999px",
                    box_shadow="0 10px 24px rgba(15, 23, 42, 0.18)",
                ),
                justify="end",
                width="100%",
            ),
            align="end",
            spacing="3",
        ),
        position="fixed",
        right="1.25rem",
        bottom="1.25rem",
        z_index="1000",
    )

# Single-column layout ensures full-width tables and charts
def main_layout() -> rx.Component:
    return rx.vstack(
        monthly_essentials_report(),
        repeated_payment_report(),
        recurring_monthly_report(),
        micro_leakage_report(),
        category_card(),
        monthly_stacked_category_chart(),
        monthly_bar_chart(),
        trend_chart(),
        monthly_summary_card(),
        top_expenses_card(),
        recurring_card(),
        forecast_card(),
        transaction_table_card(),
        spacing="6",
        width="100%",
    )

def index() -> rx.Component:
    return rx.vstack(
        navbar(),
        rx.vstack(
            upload_section(),
            dashboard_stats(),
            main_layout(),
            width="100%",
            max_width="1360px",
            padding_x="1.5em",
            spacing="5",
        ),
        payment_chat_widget(),
        padding_bottom="4em",
        width="100vw",
        min_height="100vh",
        background_color="var(--gray-1)",
    )

def details_page() -> rx.Component:
    return rx.vstack(
        navbar(),
        rx.vstack(
            rx.hstack(
                rx.button(
                    rx.icon("arrow-left", size=16),
                    "Back",
                    on_click=rx.call_script("window.history.length > 1 ? window.history.back() : window.location.assign('/')"),
                    size="3",
                    variant="soft",
                ),
                rx.button(
                    rx.icon("home", size=16),
                    "Dashboard",
                    on_click=rx.redirect("/"),
                    size="3",
                    variant="ghost",
                ),
                justify="start",
                spacing="3",
                width="100%",
            ),
            transaction_detail_panel(),
            width="100%",
            max_width="1360px",
            padding_x="1.5em",
            padding_top="1.5em",
            spacing="5",
        ),
        payment_chat_widget(),
        padding_bottom="4em",
        width="100vw",
        min_height="100vh",
        background_color="var(--gray-1)",
    )

app = rx.App()
app.add_page(index, title="Bank Statement Analyzer")
app.add_page(details_page, route="/details", title="Transaction Details")

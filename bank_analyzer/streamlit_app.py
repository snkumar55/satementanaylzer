from __future__ import annotations

import copy
import hashlib
import io
import inspect
import re
from dataclasses import MISSING

import pandas as pd
import plotly.express as px
import streamlit as st
from reflex_base.event import EventHandler

try:
    from bank_analyzer.bank_analyzer import AnalyzerState
except ImportError:
    from bank_analyzer.bank_analyzer.bank_analyzer import AnalyzerState


CURRENCY_SYMBOL = "₹"
APP_CSS = """
<style>
    .stApp {
        background: #f6f8fb;
        color: #172033;
    }
    [data-testid="stHeader"] {
        background: rgba(246, 248, 251, 0.92);
    }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 {
        color: #172033;
        letter-spacing: -0.025em;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e9f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(25, 42, 70, 0.035);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #172033;
        font-weight: 700;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-weight: 600;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e4e9f0;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
    }
    div.stButton > button, div.stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 9px;
        font-weight: 600;
    }
    [data-testid="stCaptionContainer"] {
        color: #64748b;
    }
</style>
"""


def render_brand_header():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.75rem;margin:0 0 .25rem">
          <div style="width:2.45rem;height:2.45rem;border-radius:.8rem;background:#e8f3ef;
                      color:#087e67;display:flex;align-items:center;justify-content:center;
                      font-size:1.25rem;font-weight:800">B</div>
          <div style="font-size:1.05rem;font-weight:750;letter-spacing:-.02em;color:#172033">
            Bank Analyzer
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Your money, clearly.")
    st.caption("A clear view of the activity in your bank statement.")


def format_currency(value):
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(amount):,.2f}"


def statement_records(engine):
    frame = engine._chat_dataframe()
    if frame.empty:
        return frame
    frame["DateObj"] = pd.to_datetime(frame["DateObj"], errors="coerce")
    return annotate_transfer_types(frame)


def filter_question_period(engine, frame, question):
    selected, label = engine._chat_period_filter(frame, question)
    latest_date = frame["DateObj"].max()
    relative_months = re.search(r"\b(?:last|past|previous)\s+(\d{1,2})\s+months?\b", question, re.I)
    if relative_months and not pd.isna(latest_date):
        month_count = max(1, min(24, int(relative_months.group(1))))
        latest_period = latest_date.to_period("M")
        first_period = latest_period - (month_count - 1)
        selected = frame[frame["MonthPeriod"].between(first_period, latest_period)]
        label = f"last {month_count} months"
    elif re.search(r"\b(this year|year to date|ytd)\b", question, re.I) and not pd.isna(latest_date):
        selected = frame[frame["DateObj"].dt.year == latest_date.year]
        label = f"{latest_date.year} year to date"

    days_match = re.search(r"\b(last|past|previous)\s+(\d{1,3})\s+days?\b", question, re.I)
    if days_match and not pd.isna(latest_date):
        day_count = max(1, min(366, int(days_match.group(2))))
        cutoff = latest_date - pd.Timedelta(days=day_count - 1)
        selected = frame[frame["DateObj"].between(cutoff.normalize(), latest_date)]
        label = f"last {day_count} days"

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    if year_match and not re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", question, re.I):
        selected = selected[selected["DateObj"].dt.year == int(year_match.group(1))]
        label = year_match.group(1)

    date_matches = re.findall(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", question)
    if date_matches:
        parsed_dates = [pd.to_datetime(value, errors="coerce", dayfirst="/" in value or "-" not in value) for value in date_matches]
        parsed_dates = [value for value in parsed_dates if not pd.isna(value)]
        if len(parsed_dates) >= 2:
            start_date, end_date = min(parsed_dates[:2]), max(parsed_dates[:2])
            selected = selected[selected["DateObj"].between(start_date, end_date)]
            label = f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
        elif parsed_dates:
            selected = selected[selected["DateObj"].dt.date == parsed_dates[0].date()]
            label = parsed_dates[0].strftime("%Y-%m-%d")
    return selected, label


def question_filters(engine, frame, question):
    selected, period_label = filter_question_period(engine, frame, question)
    lowered = question.lower()

    direction = None
    if (
        re.search(r"\b(deposit|deposits|deposited|credited|credits|income|received|inflow|incoming|got)\b", lowered)
        or re.search(r"\bcredit\b(?!\s+card)", lowered)
        or re.search(r"\b(money|cash)\s+in\b|\bcame in\b|\bsent me\b|\bpaid me\b|\binto my account\b", lowered)
    ):
        direction = "Deposit"
    elif re.search(
        r"\b(withdraw|withdrawn|withdrawal|withdrawals|debit|debited|payment|payments|paid|spend|spent|expense|expenses|outflow|outgoing)\b"
        r"|\b(money|cash)\s+out\b|\bwent out\b|\bsent to\b|\bpaid to\b",
        lowered,
    ):
        direction = "Withdrawal"
    if direction:
        selected = selected[selected[direction] > 0]
        if direction == "Withdrawal":
            asks_for_own_transfer = bool(re.search(r"\b(own|self)\s+accounts?\b|\btransfer(?:s)?\s+to\s+myself\b", lowered))
            selected = selected[selected["SelfTransfer"]] if asks_for_own_transfer else selected[~selected["SelfTransfer"]]

    categories = sorted(frame["Category"].dropna().astype(str).unique())
    category = next(
        (value for value in sorted(categories, key=len, reverse=True) if value.lower() in lowered),
        None,
    )
    if category:
        selected = selected[selected["Category"].str.lower() == category.lower()]

    mode_aliases = {
        "upi": "UPI", "card": "Card", "credit card": "Card", "debit card": "Card",
        "cash": "ATM/Cash", "atm": "ATM/Cash", "neft": "Transfer", "imps": "Transfer",
        "rtgs": "Transfer", "autopay": "Auto Debit", "auto debit": "Auto Debit",
    }
    selected_mode = next(
        (mode for alias, mode in sorted(mode_aliases.items(), key=lambda item: len(item[0]), reverse=True) if alias in lowered),
        None,
    )
    if selected_mode:
        selected = selected[selected["PaymentMode"].str.lower() == selected_mode.lower()]

    amount_match = re.search(
        r"\b(above|over|greater than|more than|at least|below|under|less than|at most)\s*"
        r"(?:₹|rs\.?\s*|inr\s*)?([\d,]+(?:\.\d+)?)",
        lowered,
    )
    if amount_match:
        operator, raw_amount = amount_match.groups()
        threshold = float(raw_amount.replace(",", ""))
        transaction_amount = selected[["Deposit", "Withdrawal"]].max(axis=1)
        if operator in {"above", "over", "greater than", "more than"}:
            selected = selected[transaction_amount > threshold]
        elif operator == "at least":
            selected = selected[transaction_amount >= threshold]
        elif operator in {"below", "under", "less than"}:
            selected = selected[transaction_amount < threshold]
        else:
            selected = selected[transaction_amount <= threshold]

    normalized_question = re.sub(r"[^a-z0-9 ]", " ", lowered)
    normalized_question = re.sub(r"\s+", " ", normalized_question).strip()
    merchants = frame[["Merchant", "PaymentAccountDisplay", "RepeatPaymentDisplay"]].fillna("")
    entity_candidates = set()
    for column in merchants:
        entity_candidates.update(value.strip() for value in merchants[column].astype(str) if value.strip())

    def normalize_text(value):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()

    entity = next(
        (
            value for value in sorted(entity_candidates, key=len, reverse=True)
            if normalize_text(value)
            and re.search(rf"\b{re.escape(normalize_text(value))}\b", normalized_question)
        ),
        None,
    )
    if entity:
        selected = engine._chat_match_records(selected, entity)
    return selected, period_label, direction, category, selected_mode, entity


def answer_statement_question(engine, question):
    frame = statement_records(engine)
    if frame.empty:
        return "Upload a statement first so I can answer from its transactions."

    matches, period_label, direction, category, mode, entity = question_filters(engine, frame, question)
    lowered = question.lower()
    amount_column = direction or "Amount"
    if amount_column == "Amount":
        matches = matches.copy()
        matches["_AbsoluteAmount"] = matches["Amount"].abs()
        amount_column = "_AbsoluteAmount"
    values = matches[amount_column] if not matches.empty else pd.Series(dtype=float)
    is_question_about_extreme = re.search(r"\b(highest|largest|biggest|maximum|max|lowest|smallest|minimum|min|least)\b", lowered)
    asks_for_average = re.search(r"\b(average|avg|mean)\b", lowered)
    asks_for_count = re.search(r"\b(count|how many|number of|frequency)\b", lowered)
    asks_for_top = bool(re.search(r"\b(top|most|highest|largest|biggest)\b", lowered))
    asks_for_frequency = bool(re.search(r"\b(frequent|frequently|most often|most transactions|most payments)\b", lowered))
    asks_for_merchant_rank = asks_for_frequency or (
        asks_for_top
        and bool(re.search(r"\b(merchant|merchants|payee|payees|recipient|recipients|source|sources)\b", lowered))
    )

    if "help" in lowered or "what can you" in lowered or "what questions" in lowered:
        return (
            "Ask about totals, highest or lowest transactions, averages, transaction counts, merchants, "
            "categories, payment methods, recurring payments, or monthly summaries. You can add a month, "
            "year, or date range to narrow the answer."
        )

    if matches.empty:
        return f"No matching transactions were found for {period_label}. Try broadening the date range or removing a filter."

    def matching_returned_amount():
        period_rows, _ = filter_question_period(engine, frame, question)
        returned_rows = period_rows[period_rows["ReturnedPayment"]]
        if entity:
            returned_rows = engine._chat_match_records(returned_rows, entity)
        if category:
            returned_rows = returned_rows[returned_rows["Category"].str.casefold() == category.casefold()]
        if mode:
            returned_rows = returned_rows[returned_rows["PaymentMode"].str.casefold() == mode.casefold()]
        return float(returned_rows["Deposit"].sum())

    if re.search(r"\b(recurring|repeated|repeat)\b", lowered):
        payments = matches[matches["Withdrawal"] > 0]
        grouped = (
            payments.groupby("Merchant", dropna=False)
            .agg(Transactions=("Withdrawal", "size"), Total=("Withdrawal", "sum"), Months=("MonthPeriod", "nunique"))
            .reset_index()
        )
        recurring = grouped[(grouped["Transactions"] >= 2) | (grouped["Months"] >= 2)]
        recurring = recurring.sort_values("Total", ascending=False).head(engine._chat_top_n(lowered))
        if recurring.empty:
            return f"No repeated or recurring payments were found in {period_label}."
        lines = [
            f"- {row['Merchant'] or 'Unknown'}: {int(row['Transactions'])} payments, "
            f"{int(row['Months'])} months, {format_currency(row['Total'])}"
            for _, row in recurring.iterrows()
        ]
        return f"Repeated payments in {period_label}:\n" + "\n".join(lines)

    if re.search(r"\b(latest|newest|most recent|last transaction|last payment|last deposit|last withdrawal)\b", lowered):
        row = matches.sort_values("DateObj").iloc[-1]
        movement = "deposit" if row["Deposit"] > 0 else "withdrawal"
        amount = row["Deposit"] if row["Deposit"] > 0 else row["Withdrawal"]
        name = row.get("Merchant") or row.get("Description") or "Unknown"
        return (
            f"The latest matching {movement} was {format_currency(amount)} "
            f"{'from' if movement == 'deposit' else 'to'} {name} on {row.get('Date', 'an unknown date')}."
        )

    if re.search(r"\b(monthly|each month|every month|by month|per month)\b", lowered):
        monthly = matches.groupby("MonthPeriod").agg(
            Deposits=("Deposit", "sum"),
            Withdrawals=("Withdrawal", "sum"),
            Transactions=("Date", "size"),
        ).sort_index()
        lines = [
            f"- {month.to_timestamp():%b %Y}: credited {format_currency(row['Deposits'])}, "
            f"withdrawn {format_currency(row['Withdrawals'])}, {int(row['Transactions'])} transactions"
            for month, row in monthly.iterrows()
        ]
        return "Monthly statement breakdown:\n" + "\n".join(lines)

    if re.search(r"\b(category|categories)\b", lowered):
        metric = direction or "Withdrawal"
        result_count = 1 if is_question_about_extreme else engine._chat_top_n(lowered)
        grouped = (
            matches.groupby("Category", dropna=False)[metric]
            .sum()
            .sort_values(ascending=False)
            .head(result_count)
        )
        return "Category breakdown in " + period_label + ":\n" + "\n".join(
            f"- {name or 'Other'}: {format_currency(value)}" for name, value in grouped.items()
        )

    if re.search(r"\b(payment method|payment mode|methods|modes|by mode)\b", lowered):
        metric = direction or "Withdrawal"
        grouped = (
            matches.groupby("PaymentMode", dropna=False)[metric]
            .sum()
            .sort_values(ascending=False)
            .head(engine._chat_top_n(lowered))
        )
        return "Payment method breakdown in " + period_label + ":\n" + "\n".join(
            f"- {name or 'Unknown'}: {format_currency(value)}" for name, value in grouped.items()
        )

    if is_question_about_extreme:
        order = not bool(re.search(r"\b(lowest|smallest|minimum|min|least)\b", lowered))
        row = matches.sort_values(amount_column, ascending=not order).iloc[0]
        movement = direction.lower() if direction else ("deposit" if row["Deposit"] > 0 else "withdrawal")
        label = "from" if movement == "deposit" else "to"
        payee = row.get("Merchant") or row.get("Description") or "Unknown"
        return (
            f"The {('highest' if order else 'lowest')} {movement} in {period_label} was "
            f"{format_currency(abs(float(row[amount_column])))} {label} {payee} "
            f"on {row.get('Date', 'an unknown date')}."
        )

    if asks_for_top and re.search(r"\btransactions?\b", lowered) and not asks_for_merchant_rank:
        top_rows = matches.assign(_Amount=matches[["Deposit", "Withdrawal"]].max(axis=1))
        top_rows = top_rows.sort_values("_Amount", ascending=False).head(engine._chat_top_n(lowered))
        lines = [
            f"- {row['Date']}: {row['Description'] or row['Merchant']} — {format_currency(row['_Amount'])}"
            for _, row in top_rows.iterrows()
        ]
        return f"Largest matching transactions in {period_label}:\n" + "\n".join(lines)

    if asks_for_average:
        return f"The average matching transaction in {period_label} was {format_currency(values.abs().mean())} ({len(matches)} transactions)."

    if asks_for_merchant_rank:
        ranking_column = direction or "Withdrawal"
        grouped = (
            matches.groupby("Merchant", dropna=False)
            .agg(Count=(ranking_column, "size"), Total=(ranking_column, "sum"))
            .reset_index()
            .sort_values(
                ["Count", "Total"] if asks_for_frequency else ["Total", "Count"],
                ascending=[False, False],
            )
            .head(engine._chat_top_n(lowered))
        )
        lines = [
            f"{index}. {row['Merchant'] or 'Unknown'} — {int(row['Count'])} transactions, "
            f"{format_currency(row['Total'])}"
            for index, (_, row) in enumerate(grouped.iterrows(), start=1)
        ]
        return f"Merchant ranking in {period_label}:\n" + "\n".join(lines)

    if asks_for_count:
        return f"There were {len(matches)} matching transactions in {period_label}, totaling {format_currency(values.abs().sum())}."

    wants_list = bool(re.search(r"\b(list|show|find|which|every|all transactions|transactions)\b", lowered))
    if asks_for_top and (not entity or "merchant" in lowered or "payee" in lowered or "source" in lowered):
        group_column = "Merchant"
        ranking_column = direction or "Withdrawal"
        grouped = (
            matches.groupby(group_column, dropna=False)
            .agg(Count=(ranking_column, "size"), Total=(ranking_column, "sum"))
            .reset_index()
            .sort_values("Total", ascending=False)
            .head(engine._chat_top_n(lowered))
        )
        if grouped.empty:
            return f"No matching merchants were found in {period_label}."
        lines = [
            f"{index}. {row[group_column] or 'Unknown'} — {int(row['Count'])} transactions, {format_currency(row['Total'])}"
            for index, (_, row) in enumerate(grouped.iterrows(), start=1)
        ]
        return f"Top merchants in {period_label}:\n" + "\n".join(lines)

    if wants_list:
        table = matches.sort_values("DateObj", ascending=False).head(15)
        lines = [
            f"- {row['Date']}: {row['Description'] or row['Merchant']} — "
            f"{format_currency(row['Deposit'] if row['Deposit'] > 0 else row['Withdrawal'])}"
            for _, row in table.iterrows()
        ]
        suffix = f"\nShowing 15 of {len(matches)} matching transactions." if len(matches) > 15 else ""
        return f"Matching transactions in {period_label}:\n" + "\n".join(lines) + suffix

    if entity or category or mode or direction:
        if direction:
            total = float(matches[direction].sum())
            if direction == "Withdrawal" and re.search(r"\b(own|self)\s+accounts?\b|\btransfer(?:s)?\s+to\s+myself\b", lowered):
                return (
                    f"Own-account transfers totaled {format_currency(total)} across {len(matches)} transactions "
                    f"in {period_label}. They are shown separately and excluded from spending."
                )
            label = "credited" if direction == "Deposit" else "withdrawn (excluding own-account transfers)"
            if direction == "Withdrawal":
                returned_amount = matching_returned_amount()
                net_amount = total - returned_amount
                return (
                    f"External withdrawals in {period_label}: {format_currency(total)} across {len(matches)} transactions."
                    f"\nIdentified returns: {format_currency(returned_amount)}"
                    f"\nNet spending after returns: {format_currency(net_amount)}"
                )
        else:
            total = matches["Deposit"].sum() + matches["Withdrawal"].sum()
            label = "moved"
        return (
            f"{format_currency(total)} {label} in {period_label} across {len(matches)} transactions."
            f"\nDeposits: {format_currency(matches['Deposit'].sum())}"
            f"\nWithdrawals: {format_currency(matches['Withdrawal'].sum())}"
        )

    deposits = matches[matches["Deposit"] > 0]
    withdrawals = matches[matches["Withdrawal"] > 0]
    return (
        f"Statement summary for {period_label} ({len(matches)} transactions):\n"
        f"- Total credited: {format_currency(deposits['Deposit'].sum())} across {len(deposits)} transactions\n"
        f"- Total withdrawn: {format_currency(withdrawals['Withdrawal'].sum())} across {len(withdrawals)} transactions\n"
        f"- Net cash flow: {format_currency(deposits['Deposit'].sum() - withdrawals['Withdrawal'].sum())}"
    )


def create_analyzer():
    methods = {}
    for name, member in AnalyzerState.__dict__.items():
        if inspect.isfunction(member):
            methods[name] = member
        elif isinstance(member, EventHandler):
            methods[name] = member.fn

    engine_type = type("StreamlitAnalyzer", (), methods)
    engine = engine_type()

    for name, field in AnalyzerState.get_fields().items():
        if not field.is_var:
            continue
        if field.default_factory is not None:
            value = field.default_factory()
        elif field.default is not MISSING:
            value = copy.deepcopy(field.default)
        else:
            continue
        setattr(engine, name, value)

    return engine


def load_statement(engine, uploaded_file):
    content = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(content), dtype=str)
    elif filename.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(io.BytesIO(content), dtype=str)
    else:
        raise ValueError("Upload a CSV, XLSX, or XLS statement.")

    if "Date" not in frame.columns:
        raise ValueError("The statement must contain a Date column.")

    engine.process_transactions(frame)
    if not engine.raw_transactions:
        raise ValueError("No valid transactions were found. Check the statement's dates and debit/credit columns.")


def render_overview(engine):
    records = annotate_transfer_types(pd.DataFrame(engine.raw_transactions))
    external_spend = float(records.loc[~records["SelfTransfer"], "Withdrawal"].sum())
    returned = float(records["ReturnedAmount"].sum())
    self_transfers = float(records["SelfTransferAmount"].sum())
    net_spending = external_spend - returned
    st.subheader("Overview")
    st.caption("Your statement at a glance. Spending excludes identified returns and transfers to your own account.")
    primary_columns = st.columns([1.25, 1, 1])
    primary_columns[0].metric(
        "Net spending",
        format_currency(net_spending),
        help="External withdrawals minus identified returns. Transfers to your own account are excluded.",
        border=True,
    )
    primary_columns[1].metric("Money in", engine.total_income, help="Total credits in the loaded statement.", border=True)
    primary_columns[2].metric("Money out", engine.total_expenses, help="Gross withdrawals before transfer and return adjustments.", border=True)
    st.caption("Returns and own-account transfers are inferred from the wording in your statement.")
    with st.expander("How totals are calculated"):
        reconciliation = st.columns(4)
        reconciliation[0].metric("Transactions", f"{len(records):,}")
        reconciliation[1].metric("External withdrawals", format_currency(external_spend))
        reconciliation[2].metric("Identified returns", format_currency(returned))
        reconciliation[3].metric("Own-account transfers", format_currency(self_transfers))
        st.caption(
            f"Net spending = external withdrawals ({format_currency(external_spend)}) "
            f"− identified returns ({format_currency(returned)}) = {format_currency(net_spending)}. "
            "Transfers to your own account are excluded. Return and transfer labels are inferred from statement descriptions."
        )

    monthly = pd.DataFrame(engine.monthly_chart_data)
    if not monthly.empty:
        st.subheader("Monthly cash flow")
        st.caption("Gross deposits, withdrawals, and net cash flow by month; gross withdrawals include transfers to your own account.")
        chart = monthly.set_index("month")[["deposit", "withdrawal", "savings"]]
        st.bar_chart(
            chart,
            color=["#138a72", "#d45a58", "#5377c5"],
            y_label="Amount (₹)",
            height=340,
        )

    external_withdrawals = records[(records["Withdrawal"] > 0) & ~records["SelfTransfer"]]
    category_totals = (
        external_withdrawals.groupby("Category")["Withdrawal"]
        .sum().sort_values(ascending=False).map(format_currency).reset_index()
    )
    category_totals.columns = ["Category", "External withdrawals"]
    left, right = st.columns(2)
    with left:
        st.subheader("Largest payments")
        st.caption("Highest external withdrawals in this statement.")
        largest = external_withdrawals.nlargest(10, "Withdrawal").copy()
        largest["Date"] = pd.to_datetime(largest["Date"], errors="coerce").dt.strftime("%d %b %Y")
        largest["Withdrawal"] = largest["Withdrawal"].map(format_currency)
        largest = largest.rename(columns={"PaymentMode": "Method", "Withdrawal": "Payment amount"})
        st.dataframe(
            largest[["Date", "Description", "Category", "Method", "Payment amount"]],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Spending by category")
        st.caption("External withdrawals grouped by category.")
        st.dataframe(
            category_totals,
            hide_index=True,
            width="stretch",
        )

    if not records.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Deposit sources")
            deposits = records[(records["Deposit"] > 0) & ~records["SelfTransfer"]]
            if deposits.empty:
                st.info("No external deposits were found.")
            else:
                sources = (
                    deposits.groupby("Merchant", dropna=False)
                    .agg(Transactions=("Deposit", "size"), Total=("Deposit", "sum"))
                    .sort_values("Total", ascending=False)
                    .head(10)
                )
                sources["Total"] = sources["Total"].map(format_currency)
                sources = sources.rename_axis("Source").reset_index()
                st.dataframe(sources, hide_index=True, width="stretch")
        with right:
            st.subheader("Payments by method")
            methods = (
                external_withdrawals
                .groupby("PaymentMode", dropna=False)["Withdrawal"]
                .sum()
                .sort_values(ascending=False)
            )
            if not methods.empty:
                st.bar_chart(methods, color="#138a72", y_label="Amount (₹)", height=250)

    with st.expander("More statement insights"):
        monthly_categories = pd.DataFrame(engine.monthly_stacked_category_data)
        if not monthly_categories.empty:
            st.subheader("Monthly spending by category")
            st.bar_chart(monthly_categories.set_index("Month"), y_label="Amount (₹)", height=300)

        recurring = pd.DataFrame(
            engine.monthly_recurring_table,
            columns=["Merchant", "Category", "Mode", "Months paid", "Average per month", "Total paid", "Monthly range", "Latest month"],
        )
        if not recurring.empty:
            st.subheader("Recurring payments")
            st.caption("Payments to the same merchant found in multiple statement months.")
            st.dataframe(recurring, hide_index=True, width="stretch")

        anomalies = records[records["Anomaly"].astype(str).str.contains("Outlier", case=False, na=False)]
        if not anomalies.empty:
            st.subheader("Transactions to review")
            st.caption("Statistical outliers only; an unusual amount is not necessarily an error.")
            anomalies = anomalies.copy()
            anomalies["Withdrawal"] = anomalies["Withdrawal"].map(format_currency)
            anomalies["Deposit"] = anomalies["Deposit"].map(format_currency)
            st.dataframe(
                anomalies[["Date", "Description", "Withdrawal", "Deposit"]],
                hide_index=True,
                width="stretch",
            )

        if engine.forecast_table:
            st.subheader("Next-month estimate")
            st.caption("A trend-based estimate from historical monthly totals, not a guarantee.")
            forecast = pd.DataFrame(
                engine.forecast_table,
                columns=["Period", "Deposit", "Withdrawal", "Savings"],
            )
            for column in ("Deposit", "Withdrawal", "Savings"):
                forecast[column] = forecast[column].apply(
                    lambda value: value if value == "N/A" else format_currency(float(str(value).replace("₹", "").replace(",", "")))
                )
            st.dataframe(forecast, hide_index=True, width="stretch")


def annotate_transfer_types(records):
    annotated = records.copy()
    for column in ("Description", "Merchant", "PaymentAccountDisplay", "PaymentMode", "Category"):
        if column not in annotated:
            annotated[column] = ""
        annotated[column] = annotated[column].fillna("").astype(str)

    narration = (
        annotated["Description"] + " " + annotated["Merchant"] + " "
        + annotated["PaymentAccountDisplay"]
    ).str.casefold()
    annotated["SelfTransfer"] = narration.str.contains(
        r"\bsharath\b|\bself(?:\s|-)?transfer\b|\bown\s+account\b|\bbetween\s+(?:my|own)\s+accounts?\b",
        regex=True,
        na=False,
    )
    annotated["ReturnedPayment"] = (
        (annotated["Deposit"] > 0)
        & ~annotated["SelfTransfer"]
        & narration.str.contains(r"\b(?:refund|refunded|return(?:ed)?|reversal|reversed|chargeback|re-credited)\b", regex=True, na=False)
    )
    transfer_signals = narration.str.contains(
        r"\b(?:transfer|transferred|sent|imps|neft|rtgs|p2p|p2a|beneficiary)\b",
        regex=True,
        na=False,
    )
    recipient_signal = narration.str.contains(r"\bto\s+[a-z][a-z0-9._-]*\b", regex=True, na=False)
    annotated["SentToOthers"] = (
        (annotated["Withdrawal"] > 0)
        & ~annotated["SelfTransfer"]
        & annotated["PaymentMode"].str.contains(r"UPI|Transfer", case=False, regex=True)
        & (transfer_signals | (recipient_signal & annotated["Category"].eq("Other")))
    )
    merchant_key = (
        annotated["Merchant"]
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    for index, row in annotated[annotated["Deposit"] > 0].iterrows():
        if row["SelfTransfer"] or annotated.at[index, "ReturnedPayment"]:
            continue
        prior_sent = annotated[
            annotated["SentToOthers"]
            & (annotated["Date"] < row["Date"])
            & ((annotated["Withdrawal"] - row["Deposit"]).abs() < 0.01)
            & merchant_key.eq(merchant_key.at[index])
            & merchant_key.ne("")
            & merchant_key.ne("unknown")
        ]
        if not prior_sent.empty:
            annotated.at[index, "ReturnedPayment"] = True
    annotated["TransactionType"] = "Payment"
    annotated.loc[annotated["Deposit"] > 0, "TransactionType"] = "Deposit"
    annotated.loc[annotated["SentToOthers"], "TransactionType"] = "Sent to someone"
    annotated.loc[annotated["ReturnedPayment"], "TransactionType"] = "Money returned"
    annotated.loc[annotated["SelfTransfer"], "TransactionType"] = "Own-account transfer"
    annotated["ReturnedAmount"] = annotated["Deposit"].where(annotated["ReturnedPayment"], 0.0)
    annotated["SelfTransferAmount"] = annotated[["Deposit", "Withdrawal"]].max(axis=1).where(
        annotated["SelfTransfer"], 0.0
    )
    return annotated


def spending_intelligence_data(engine):
    records = pd.DataFrame(engine.raw_transactions)
    if records.empty:
        return records, records

    records["Date"] = pd.to_datetime(records["Date"], errors="coerce")
    for column in ("Withdrawal", "Deposit", "Amount"):
        records[column] = pd.to_numeric(records[column], errors="coerce").fillna(0.0)
    records = records.dropna(subset=["Date"])
    records = annotate_transfer_types(records)

    spending = records[(records["Withdrawal"] > 0) & ~records["SelfTransfer"]].copy()
    spending["MonthPeriod"] = spending["Date"].dt.to_period("M")
    spending["Month"] = spending["MonthPeriod"].dt.to_timestamp()
    spending["Merchant"] = spending["Merchant"].replace("", "Unknown").fillna("Unknown")
    spending["Category"] = spending["Category"].replace("", "Other").fillna("Other")
    return records, spending


def spending_health_score(spending, category_totals, merchant_totals, monthly):
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


def render_spending_intelligence(engine):
    records, spending = spending_intelligence_data(engine)
    st.subheader("Spending Intelligence")
    st.caption(
        "A focused view of external debit transactions. Transfers to your own account are excluded; "
        "refund labels are inferred from statement descriptions."
    )
    if spending.empty:
        st.info("No external debit transactions are available to analyze in this statement.")
        return

    total_spending = float(spending["Withdrawal"].sum())
    category_totals = (
        spending.groupby("Category", dropna=False)
        .agg(Total=("Withdrawal", "sum"), Transactions=("Withdrawal", "size"))
        .sort_values("Total", ascending=False)
    )
    category_totals["Share"] = category_totals["Total"] / total_spending

    merchant_totals = (
        spending.groupby("Merchant", dropna=False)
        .agg(
            Transactions=("Withdrawal", "size"),
            Total=("Withdrawal", "sum"),
            Average=("Withdrawal", "mean"),
        )
        .sort_values("Total", ascending=False)
    )
    monthly = (
        spending.groupby("MonthPeriod")
        .agg(Total=("Withdrawal", "sum"), Transactions=("Withdrawal", "size"))
        .sort_index()
    )
    month_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(month_range, fill_value=0)
    monthly.index = monthly.index.to_timestamp()
    monthly.index.name = "Month"
    previous_month = monthly["Total"].shift(1)
    monthly["ChangePct"] = ((monthly["Total"] - previous_month) / previous_month.where(previous_month > 0)) * 100

    category_rows = category_totals.reset_index().rename(columns={"index": "Category"})
    category_rows["Total Amount Spent"] = category_rows["Total"].map(format_currency)
    category_rows["% of Total Spending"] = category_rows["Share"].map(lambda value: f"{value:.1%}")
    category_rows = category_rows[["Category", "Total Amount Spent", "% of Total Spending"]]

    merchant_rows = merchant_totals.head(20).reset_index()
    merchant_rows["Total Amount Spent"] = merchant_rows["Total"].map(format_currency)
    merchant_rows["Average Transaction Value"] = merchant_rows["Average"].map(format_currency)
    merchant_rows = merchant_rows.rename(columns={"Transactions": "Transaction Count"})
    merchant_rows = merchant_rows[
        ["Merchant", "Transaction Count", "Total Amount Spent", "Average Transaction Value"]
    ]

    score, score_drivers, score_components = spending_health_score(
        spending, category_totals, merchant_totals, monthly
    )
    kpis = st.columns(4)
    kpis[0].metric("External spending", format_currency(total_spending), border=True)
    kpis[1].metric("Transactions", f"{len(spending):,}", border=True)
    kpis[2].metric("Average transaction", format_currency(spending["Withdrawal"].mean()), border=True)
    kpis[3].metric("Average per calendar day", format_currency(total_spending / max(1, (spending["Date"].max().date() - spending["Date"].min().date()).days + 1)), border=True)

    st.subheader("Top spending categories")
    category_chart, category_pie = st.columns([1.35, 1])
    with category_chart:
        bar = px.bar(
            category_totals.reset_index(),
            x="Total",
            y="Category",
            orientation="h",
            color_discrete_sequence=["#138a72"],
            labels={"Total": "Amount spent (₹)", "Category": ""},
        )
        bar.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"}, margin=dict(l=8, r=12, t=12, b=8), height=350)
        bar.update_traces(hovertemplate="%{y}<br>₹%{x:,.2f}<extra></extra>")
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
    with category_pie:
        pie = px.pie(
            category_totals.reset_index(),
            names="Category",
            values="Total",
            hole=0.62,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        pie.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=12, b=8), height=350, legend_title_text="")
        pie.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}<br>₹%{value:,.2f} (%{percent})<extra></extra>",
        )
        st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
    st.dataframe(category_rows, hide_index=True, width="stretch")

    lead_category = category_totals.index[0]
    lead_share = float(category_totals.iloc[0]["Share"])
    category_insights = [f"{lead_share:.0%} of external spending was in {lead_category}."]
    if len(category_totals) > 1:
        category_insights.append(f"{category_totals.index[1]} was the second-largest spending category.")
    st.caption(" ".join(category_insights))

    st.subheader("Top merchants")
    st.dataframe(merchant_rows, hide_index=True, width="stretch")
    leading_merchant = merchant_totals.iloc[0]
    st.caption(
        f"{merchant_totals.index[0]} accounts for {format_currency(leading_merchant['Total'])} "
        f"across {int(leading_merchant['Transactions'])} transactions."
    )

    st.subheader("Spending concentration")
    top_five_category_share = float(category_totals.head(5)["Total"].sum() / total_spending)
    top_ten_merchant_share = float(merchant_totals.head(10)["Total"].sum() / total_spending)
    concentration_metrics = st.columns(2)
    concentration_metrics[0].metric("Top 5 categories", f"{top_five_category_share:.1%} of spending", border=True)
    concentration_metrics[1].metric("Top 10 merchants", f"{top_ten_merchant_share:.1%} of spending", border=True)
    categories_to_65 = int((category_totals["Share"].cumsum() < 0.65).sum()) + 1
    categories_to_65 = min(categories_to_65, len(category_totals))
    concentration_note = (
        f"{category_totals.head(categories_to_65)['Share'].sum():.0%} of spending comes from "
        f"{categories_to_65} categories."
    )
    if top_five_category_share >= 0.65:
        concentration_note += " Spending is concentrated in a small number of categories."
    elif top_ten_merchant_share >= 0.65:
        concentration_note += " A large share is concentrated among a small group of merchants."
    else:
        concentration_note += " Spending is relatively distributed across categories and merchants."
    st.caption(concentration_note)

    st.subheader("Monthly spending trend")
    monthly_chart = px.line(
        monthly.reset_index(),
        x="Month",
        y="Total",
        markers=True,
        color_discrete_sequence=["#138a72"],
        labels={"Total": "Amount spent (₹)", "Month": ""},
    )
    monthly_chart.update_layout(template="plotly_white", margin=dict(l=8, r=12, t=12, b=8), height=340)
    monthly_chart.update_traces(hovertemplate="%{x|%b %Y}<br>₹%{y:,.2f}<extra></extra>")
    st.plotly_chart(monthly_chart, width="stretch", config={"displayModeBar": False})

    monthly_summary = monthly.copy()
    monthly_summary["Monthly spending"] = monthly_summary["Total"].map(format_currency)
    monthly_summary["Month-over-month change"] = monthly_summary["ChangePct"].map(
        lambda value: "New month" if pd.isna(value) else f"{value:+.1f}%"
    )
    monthly_summary.index = monthly_summary.index.strftime("%b %Y")
    st.dataframe(
        monthly_summary[["Monthly spending", "Month-over-month change"]].rename_axis("Month").reset_index(),
        width="stretch",
    )
    highest_month = monthly["Total"].idxmax()
    lowest_month = monthly["Total"].idxmin()
    monthly_note = (
        f"Spending was highest in {highest_month:%b %Y} ({format_currency(monthly.loc[highest_month, 'Total'])}) "
        f"and lowest in {lowest_month:%b %Y} ({format_currency(monthly.loc[lowest_month, 'Total'])})."
    )
    valid_changes = monthly["ChangePct"].dropna()
    if not valid_changes.empty:
        change = float(valid_changes.iloc[-1])
        latest_month = monthly.index[-1]
        monthly_note += (
            f" {latest_month:%b %Y} spending "
            f"{'rose' if change >= 0 else 'fell'} {abs(change):.1f}% from the previous month."
        )
    st.caption(monthly_note)

    st.subheader("Biggest expenses")
    largest_expenses = spending.nlargest(20, "Withdrawal").copy()
    largest_expenses["Date"] = largest_expenses["Date"].dt.strftime("%d %b %Y")
    largest_expenses["Amount"] = largest_expenses["Withdrawal"].map(format_currency)
    st.dataframe(
        largest_expenses[["Date", "Merchant", "Category", "Amount"]],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Spending behaviour")
    most_frequent = merchant_totals.sort_values(["Transactions", "Total"], ascending=False).iloc[0]
    largest_expense = spending.loc[spending["Withdrawal"].idxmax()]
    weekday_spend = float(spending.loc[spending["Date"].dt.dayofweek < 5, "Withdrawal"].sum())
    weekend_spend = float(spending.loc[spending["Date"].dt.dayofweek >= 5, "Withdrawal"].sum())
    active_days = max(1, (spending["Date"].max().date() - spending["Date"].min().date()).days + 1)
    daily_average = total_spending / active_days
    behaviour = [
        ("Largest category", f"{lead_category} · {lead_share:.0%} of spending"),
        ("Fastest-growing category", "Not enough month history" if len(monthly) < 2 else "See category trend below"),
        ("Most frequent merchant", f"{merchant_totals.sort_values(['Transactions', 'Total'], ascending=False).index[0]} · {int(most_frequent['Transactions'])} transactions"),
        ("Largest purchase", f"{format_currency(largest_expense['Withdrawal'])} · {largest_expense['Merchant']}"),
        ("Weekday vs weekend", f"{format_currency(weekday_spend)} weekdays · {format_currency(weekend_spend)} weekends"),
        ("Average daily spending", format_currency(daily_average)),
        ("Average transaction", format_currency(spending["Withdrawal"].mean())),
    ]
    if len(monthly) >= 2:
        category_monthly = spending.groupby(["MonthPeriod", "Category"])["Withdrawal"].sum().unstack(fill_value=0).sort_index()
        if len(category_monthly) >= 2:
            category_month_range = pd.period_range(category_monthly.index.min(), category_monthly.index.max(), freq="M")
            category_monthly = category_monthly.reindex(category_month_range, fill_value=0).sort_index()
            previous, latest = category_monthly.iloc[-2], category_monthly.iloc[-1]
            growth = ((latest - previous) / previous.where(previous > 0)).replace([float("inf"), -float("inf")], pd.NA)
            growing = growth[growth > 0]
            if not growing.empty:
                fastest = growing.idxmax()
                behaviour[1] = (
                    "Fastest-growing category",
                    f"{fastest} · +{growing[fastest]:.0%} month over month",
                )
            else:
                behaviour[1] = ("Fastest-growing category", "No category grew in the latest month")
    for start in range(0, len(behaviour), 3):
        insight_columns = st.columns(3)
        for column, (title, value) in zip(insight_columns, behaviour[start:start + 3]):
            with column:
                st.markdown(f"**{title}**")
                st.container(border=True).markdown(value)
    st.caption(
        f"Weekday and weekend figures cover the statement's recorded spending dates. "
        f"Average daily spending uses the {active_days} calendar days between the first and last debit."
    )

    st.subheader("Category deep dive")
    selected_category = st.selectbox("Choose a category", category_totals.index.tolist(), key="spending_intelligence_category")
    category_spend = spending[spending["Category"] == selected_category].copy()
    category_monthly = (
        category_spend.groupby("MonthPeriod")["Withdrawal"].sum().sort_index()
    )
    full_category_months = pd.period_range(monthly.index.min().to_period("M"), monthly.index.max().to_period("M"), freq="M")
    category_monthly = category_monthly.reindex(full_category_months, fill_value=0)
    category_merchants = (
        category_spend.groupby("Merchant")
        .agg(Transactions=("Withdrawal", "size"), Total=("Withdrawal", "sum"), Average=("Withdrawal", "mean"))
        .sort_values("Total", ascending=False)
        .head(10)
    )
    category_kpis = st.columns(3)
    category_kpis[0].metric("Total spent", format_currency(category_spend["Withdrawal"].sum()), border=True)
    category_kpis[1].metric("Transactions", f"{len(category_spend):,}", border=True)
    category_kpis[2].metric("Average transaction", format_currency(category_spend["Withdrawal"].mean()), border=True)
    deep_left, deep_right = st.columns(2)
    with deep_left:
        st.markdown("**Monthly trend**")
        if category_monthly.empty:
            st.caption("No monthly transactions for this category.")
        else:
            deep_chart_data = category_monthly.rename("Total").rename_axis("Month").reset_index()
            deep_chart_data["Month"] = deep_chart_data["Month"].dt.to_timestamp()
            deep_chart = px.line(
                deep_chart_data,
                x="Month",
                y="Total",
                markers=True,
                color_discrete_sequence=["#138a72"],
                labels={"Total": "Amount spent (₹)", "Month": ""},
            )
            deep_chart.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=8, b=8), height=300)
            deep_chart.update_traces(hovertemplate="%{x|%b %Y}<br>₹%{y:,.2f}<extra></extra>")
            st.plotly_chart(deep_chart, width="stretch", config={"displayModeBar": False})
    with deep_right:
        st.markdown("**Top merchants**")
        deep_merchants = category_merchants.reset_index()
        deep_merchants["Total spent"] = deep_merchants["Total"].map(format_currency)
        deep_merchants["Average"] = deep_merchants["Average"].map(format_currency)
        deep_merchants = deep_merchants.rename(columns={"Transactions": "Count"})
        st.dataframe(
            deep_merchants[["Merchant", "Count", "Total spent", "Average"]],
            hide_index=True,
            width="stretch",
        )
    category_fastest = behaviour[1][1] if behaviour[1][0] == "Fastest-growing category" and behaviour[1][1].startswith(selected_category) else ""
    st.caption(
        f"{selected_category} accounts for {category_totals.loc[selected_category, 'Share']:.1%} "
        f"of external spending. {category_fastest}".strip()
    )

    st.subheader("Spending score")
    score_columns = st.columns([1, 2])
    score_columns[0].metric(
        "Spending balance score",
        f"{score} / 100",
        help="An exploratory indicator based only on this statement, not financial advice or a credit score.",
        border=True,
    )
    score_columns[1].caption(
        "Higher means spending is more distributed, has fewer large purchases and recurring patterns, "
        "and varies less month to month. This is a simple statement-based indicator, not a credit score."
    )
    score_chart = px.bar(
        x=list(score_components.values()),
        y=list(score_components.keys()),
        orientation="h",
        labels={"x": "Points", "y": ""},
        range_x=[0, 25],
        color_discrete_sequence=["#138a72"],
    )
    score_chart.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=8, b=8), height=240)
    st.plotly_chart(score_chart, width="stretch", config={"displayModeBar": False})
    st.markdown("**Score drivers**")
    for driver in score_drivers:
        st.caption(f"• {driver}")


def render_transactions(engine):
    if not engine.raw_transactions:
        return

    records = pd.DataFrame(engine.raw_transactions)
    records["Date"] = pd.to_datetime(records["Date"], errors="coerce")
    for column in ("Withdrawal", "Deposit", "Amount"):
        records[column] = pd.to_numeric(records[column], errors="coerce").fillna(0.0)
    records = records.dropna(subset=["Date"])
    records = annotate_transfer_types(records)
    min_date = records["Date"].min().date()
    max_date = records["Date"].max().date()

    st.subheader("Transactions")
    search_column, type_column = st.columns([3, 1])
    keyword = search_column.text_input(
        "Search transactions",
        placeholder="Try: Swiggy, UPI, salary, Food",
        help="Search the description, payee, category, or payment method. Every search word must match.",
    )
    transaction_type = type_column.selectbox(
        "Transaction type",
        ["All", "Deposits", "Withdrawals", "Sent to someone", "Money returned", "Own-account transfers"],
    )
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, (tuple, list)):
        start_date = selected_dates[0] if selected_dates else min_date
        end_date = selected_dates[-1] if len(selected_dates) > 1 else start_date
    else:
        start_date = end_date = selected_dates

    amount_min = float(records[["Deposit", "Withdrawal"]].max(axis=1).min())
    amount_max = float(records[["Deposit", "Withdrawal"]].max(axis=1).max())
    with st.expander("Filter by amount (optional)"):
        if amount_min < amount_max:
            amount_range = st.slider(
                "Transaction amount",
                min_value=amount_min,
                max_value=amount_max,
                value=(amount_min, amount_max),
                step=max(1.0, round((amount_max - amount_min) / 100, 2)),
                format="₹%.2f",
            )
        else:
            amount_range = (amount_min, amount_max)
            st.caption(f"All loaded transactions are {format_currency(amount_min)}.")

    filtered = records[records["Date"].dt.date.between(start_date, end_date)].copy()
    if transaction_type == "Deposits":
        filtered = filtered[filtered["Deposit"] > 0]
    elif transaction_type == "Withdrawals":
        filtered = filtered[filtered["Withdrawal"] > 0]
    elif transaction_type == "Sent to someone":
        filtered = filtered[filtered["SentToOthers"]]
    elif transaction_type == "Money returned":
        filtered = filtered[filtered["ReturnedPayment"]]
    elif transaction_type == "Own-account transfers":
        filtered = filtered[filtered["SelfTransfer"]]

    row_amount = filtered[["Deposit", "Withdrawal"]].max(axis=1)
    filtered = filtered[row_amount.between(amount_range[0], amount_range[1])]

    search_terms = re.findall(r"[a-z0-9]+", keyword.casefold())
    if search_terms:
        searchable = (
            filtered[["Description", "Merchant", "PaymentAccountDisplay", "Category", "PaymentMode", "TransactionType"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.casefold()
            .str.replace(r"[^a-z0-9]+", " ", regex=True)
        )
        for term in search_terms:
            filtered = filtered[searchable.loc[filtered.index].str.contains(re.escape(term), regex=True)]

    filtered = filtered.sort_values("Date", ascending=False)
    deposits = float(filtered["Deposit"].sum())
    withdrawals = float(filtered["Withdrawal"].sum())
    returned = float(filtered["ReturnedAmount"].sum())
    self_transfers = float(filtered["SelfTransferAmount"].sum())
    spend_before_returns = float(filtered.loc[~filtered["SelfTransfer"], "Withdrawal"].sum())
    net_spending = spend_before_returns - returned
    sent_to_others = float(filtered.loc[filtered["SentToOthers"], "Withdrawal"].sum())
    summary_columns = st.columns(4)
    summary_columns[0].metric("Matching transactions", f"{len(filtered):,}", border=True)
    summary_columns[1].metric("Money in", format_currency(deposits), border=True)
    summary_columns[2].metric("Money out", format_currency(withdrawals), border=True)
    summary_columns[3].metric("Net spending", format_currency(net_spending), border=True)
    with st.expander("How these filtered totals are calculated"):
        st.caption(
            f"Applied filters: {start_date:%d %b %Y}–{end_date:%d %b %Y} · "
            f"{transaction_type} · {len(filtered):,} matching transaction(s) · "
            f"amount {format_currency(amount_range[0])}–{format_currency(amount_range[1])}."
        )
        reconciliation = st.columns(3)
        reconciliation[0].metric("Sent to others", format_currency(sent_to_others))
        reconciliation[1].metric("Identified returns", format_currency(returned))
        reconciliation[2].metric("Own-account transfers", format_currency(self_transfers))
        st.caption(
            f"Net spending = withdrawals excluding own-account transfers "
            f"({format_currency(spend_before_returns)}) − identified returns "
            f"({format_currency(returned)}) = {format_currency(net_spending)}. "
            "Transfer and return labels are inferred from statement descriptions."
        )

    display = filtered.copy()
    display["Date"] = display["Date"].dt.strftime("%d %b %Y")
    for column in ("Withdrawal", "Deposit", "Amount"):
        display[column] = display[column].map(format_currency)
    display = display.rename(
        columns={
            "PaymentAccountDisplay": "Account/Payee",
            "PaymentMode": "Mode",
            "SpendBand": "Spend band",
            "TransactionType": "Transaction type",
            "SelfTransfer": "Own-account transfer",
            "ReturnedPayment": "Returned payment",
            "SentToOthers": "Sent to someone",
            "ReturnedAmount": "Returned amount",
            "SelfTransferAmount": "Own-account transfer amount",
        }
    )
    if display.empty:
        st.info("No transactions match these filters. Try another keyword or a wider date or amount range.")
    else:
        st.dataframe(
            display[
                [
                    "Date", "Description", "Account/Payee", "Transaction type", "Category",
                    "Mode", "Withdrawal", "Deposit",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
    st.download_button(
        "Download filtered transactions as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_transactions.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )


def render_chat(engine):
    st.subheader("Ask about this statement")
    for speaker, message in engine.chat_messages:
        with st.chat_message("assistant" if speaker == "assistant" else "user"):
            st.markdown(message)

    with st.form("payment_chat_form", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="What is the highest amount credited?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        messages = list(engine.chat_messages)
        messages.extend([
            ["user", question.strip()],
            ["assistant", answer_statement_question(engine, question.strip())],
        ])
        engine.chat_messages = messages[-18:]
        st.rerun()


def main():
    st.set_page_config(page_title="Bank Statement Analyzer", page_icon="💳", layout="wide")
    render_brand_header()

    if "analyzer_engine" not in st.session_state:
        st.session_state.analyzer_engine = create_analyzer()
    engine = st.session_state.analyzer_engine

    with st.container(border=True):
        upload_columns = st.columns([1.5, 2])
        with upload_columns[0]:
            st.subheader("Start with a statement")
            st.caption("Upload a CSV or Excel file to explore your transactions.")
        with upload_columns[1]:
            uploaded_file = st.file_uploader(
                "Choose a bank statement",
                type=["csv", "xlsx", "xls"],
                help="Supported formats: CSV, XLSX, and XLS. The file is processed for this session.",
            )
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        signature = (uploaded_file.name, hashlib.sha256(content).hexdigest())
        if st.session_state.get("uploaded_signature") != signature:
            try:
                engine.raw_transactions = []
                load_statement(engine, uploaded_file)
                st.session_state.uploaded_signature = signature
                st.session_state.uploaded_filename = uploaded_file.name
                engine.chat_messages = [["assistant", "Statement loaded. Ask a question about its transactions."]]
                st.success(f"Statement ready · {uploaded_file.name} · {len(engine.raw_transactions):,} transactions")
            except (ValueError, pd.errors.ParserError, ImportError) as error:
                st.error(str(error))
            except Exception as error:
                print(f"Statement processing failed for {uploaded_file.name}: {error}")
                st.error("We couldn't read this statement. Check its date and deposit/withdrawal columns, then try again.")

    if not engine.raw_transactions:
        st.info("Your overview, transaction search, and statement Q&A will appear here after upload.")
        return

    loaded_name = st.session_state.get("uploaded_filename")
    if loaded_name:
        st.caption(f"Currently analyzing **{loaded_name}** · {len(engine.raw_transactions):,} transactions")

    overview_tab, transactions_tab, chat_tab, intelligence_tab = st.tabs(
        ["Overview", "Transactions", "Chat", "Spending Intelligence"]
    )
    with overview_tab:
        render_overview(engine)
    with transactions_tab:
        render_transactions(engine)
    with chat_tab:
        render_chat(engine)
    with intelligence_tab:
        render_spending_intelligence(engine)


if __name__ == "__main__":
    main()

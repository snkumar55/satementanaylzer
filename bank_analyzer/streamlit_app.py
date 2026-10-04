from __future__ import annotations

import copy
import hashlib
import io
import inspect
import re
from dataclasses import MISSING

import pandas as pd
import streamlit as st
from reflex_base.event import EventHandler

try:
    from bank_analyzer.bank_analyzer import AnalyzerState
except ImportError:
    from bank_analyzer.bank_analyzer.bank_analyzer import AnalyzerState


CURRENCY_SYMBOL = "₹"


def format_currency(value):
    amount = float(value)
    sign = "-" if amount < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(amount):,.2f}"


def statement_records(engine):
    frame = engine._chat_dataframe()
    if frame.empty:
        return frame
    frame["DateObj"] = pd.to_datetime(frame["DateObj"], errors="coerce")
    return frame


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
            total = matches[direction].sum()
            label = "credited" if direction == "Deposit" else "withdrawn"
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
    records = pd.DataFrame(engine.raw_transactions)
    st.subheader("Statement overview")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total credited", engine.total_income)
    metric_columns[1].metric("Total withdrawn", engine.total_expenses)
    metric_columns[2].metric("Net balance", engine.net_balance)
    metric_columns[3].metric("Transactions", f"{len(records):,}")
    metric_columns[4].metric(
        "Average transaction",
        format_currency(records["Amount"].abs().mean()) if not records.empty else format_currency(0),
    )

    monthly = pd.DataFrame(engine.monthly_chart_data)
    if not monthly.empty:
        st.subheader("Monthly deposits, withdrawals, and savings")
        chart = monthly.set_index("month")[["deposit", "withdrawal", "savings"]]
        st.bar_chart(chart)
        st.dataframe(
            pd.DataFrame(engine.monthly_summary_table, columns=["Month", "Deposit", "Withdrawal", "Savings"]),
            hide_index=True,
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Largest withdrawals")
        st.dataframe(
            pd.DataFrame(
                engine.top_expenses_table,
                columns=["Date", "Description", "Category", "Mode", "Band", "Amount"],
            ),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Withdrawals by category")
        st.dataframe(
            pd.DataFrame(engine.category_summary_table, columns=["Category", "Total Withdrawals"]),
            hide_index=True,
            width="stretch",
        )

    if not records.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Deposit sources")
            deposits = records[records["Deposit"] > 0]
            if deposits.empty:
                st.info("No deposits were found.")
            else:
                sources = (
                    deposits.groupby("Merchant", dropna=False)
                    .agg(Transactions=("Deposit", "size"), Total=("Deposit", "sum"))
                    .sort_values("Total", ascending=False)
                    .head(10)
                )
                st.dataframe(sources, width="stretch")
        with right:
            st.subheader("Payment methods")
            methods = (
                records[records["Withdrawal"] > 0]
                .groupby("PaymentMode", dropna=False)["Withdrawal"]
                .sum()
                .sort_values(ascending=False)
            )
            if not methods.empty:
                st.bar_chart(methods)

        monthly_categories = pd.DataFrame(engine.monthly_stacked_category_data)
        if not monthly_categories.empty:
            st.subheader("Monthly withdrawals by category")
            st.bar_chart(monthly_categories.set_index("Month"))

        recurring = pd.DataFrame(
            engine.monthly_recurring_table,
            columns=["Merchant", "Category", "Mode", "Months paid", "Average per month", "Total paid", "Monthly range", "Latest month"],
        )
        if not recurring.empty:
            st.subheader("Recurring payments")
            st.dataframe(recurring, hide_index=True, width="stretch")

        anomalies = records[records["Anomaly"].astype(str).str.contains("Outlier", case=False, na=False)]
        if not anomalies.empty:
            st.subheader("Unusual transactions")
            st.dataframe(
                anomalies[["Date", "Description", "Withdrawal", "Deposit", "Anomaly"]],
                hide_index=True,
                width="stretch",
            )

    if engine.forecast_table:
        st.subheader("Next-month forecast")
        st.dataframe(
            pd.DataFrame(
                engine.forecast_table,
                columns=["Period", "Deposit (forecast)", "Withdrawal (forecast)", "Savings (forecast)"],
            ),
            hide_index=True,
            width="stretch",
        )


def render_transactions(engine):
    if not engine.raw_transactions:
        return

    records = pd.DataFrame(engine.raw_transactions)
    records["Date"] = pd.to_datetime(records["Date"], errors="coerce")
    for column in ("Withdrawal", "Deposit", "Amount"):
        records[column] = pd.to_numeric(records[column], errors="coerce").fillna(0.0)
    records = records.dropna(subset=["Date"])
    min_date = records["Date"].min().date()
    max_date = records["Date"].max().date()

    st.subheader("Find transactions")
    search_column, type_column = st.columns([3, 1])
    keyword = search_column.text_input(
        "Search statement",
        placeholder="Try: Swiggy, UPI, salary, Food",
        help="Search description, merchant/payee, category, and payment method. Separate words to require all of them.",
    )
    transaction_type = type_column.selectbox("Type", ["All", "Deposits", "Withdrawals"])
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

    filtered = records[records["Date"].dt.date.between(start_date, end_date)].copy()
    if transaction_type == "Deposits":
        filtered = filtered[filtered["Deposit"] > 0]
    elif transaction_type == "Withdrawals":
        filtered = filtered[filtered["Withdrawal"] > 0]

    search_terms = re.findall(r"[a-z0-9]+", keyword.casefold())
    if search_terms:
        searchable = (
            filtered[["Description", "Merchant", "PaymentAccountDisplay", "Category", "PaymentMode"]]
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
    net = deposits - withdrawals
    summary_columns = st.columns(4)
    summary_columns[0].metric("Matching transactions", f"{len(filtered):,}")
    summary_columns[1].metric("Deposits", format_currency(deposits))
    summary_columns[2].metric("Withdrawals", format_currency(withdrawals))
    summary_columns[3].metric("Net", format_currency(net))
    st.caption(
        f"Calculation for {start_date:%d %b %Y}–{end_date:%d %b %Y}: "
        f"deposits = sum of Deposit ({format_currency(deposits)}); "
        f"withdrawals = sum of Withdrawal ({format_currency(withdrawals)}); "
        f"net = deposits − withdrawals = {format_currency(net)}. "
        f"All totals use the {len(filtered):,} matching transaction(s) shown below."
    )

    display = filtered.copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    for column in ("Withdrawal", "Deposit", "Amount"):
        display[column] = display[column].map(format_currency)
    display = display.rename(
        columns={
            "PaymentAccountDisplay": "Account/Payee",
            "PaymentMode": "Mode",
            "SpendBand": "Spend band",
        }
    )
    st.dataframe(
        display[
            ["Date", "Description", "Account/Payee", "Category", "Mode", "Spend band", "Withdrawal", "Deposit", "Amount", "Anomaly"]
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
    st.title("Bank Statement Analyzer")
    st.caption("Analyze a bank statement locally in this app. Uploaded statement data is processed in the current session.")

    if "analyzer_engine" not in st.session_state:
        st.session_state.analyzer_engine = create_analyzer()
    engine = st.session_state.analyzer_engine

    uploaded_file = st.file_uploader("Upload a bank statement", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        signature = (uploaded_file.name, hashlib.sha256(content).hexdigest())
        if st.session_state.get("uploaded_signature") != signature:
            try:
                engine.raw_transactions = []
                load_statement(engine, uploaded_file)
                st.session_state.uploaded_signature = signature
                engine.chat_messages = [["assistant", "Statement loaded. Ask a question about its transactions."]]
                st.success(f"Loaded {len(engine.raw_transactions)} transactions from {uploaded_file.name}.")
            except (ValueError, pd.errors.ParserError, ImportError) as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not process the uploaded statement: {error}")

    if not engine.raw_transactions:
        st.info("Upload a CSV or Excel statement to view the analysis and ask questions.")
        return

    overview_tab, transactions_tab, chat_tab = st.tabs(["Overview", "Transactions", "Chat"])
    with overview_tab:
        render_overview(engine)
    with transactions_tab:
        render_transactions(engine)
    with chat_tab:
        render_chat(engine)


if __name__ == "__main__":
    main()

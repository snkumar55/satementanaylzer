from __future__ import annotations

import re
import pandas as pd

from bank_analyzer.services.transfer_service import annotate_transfer_types
from bank_analyzer.utils.formatting import format_currency

def statement_records(engine) -> pd.DataFrame:
    """Build the statement dataframe used by the local Q&A service."""
    frame = engine._chat_dataframe()
    if frame.empty:
        return frame
    frame["DateObj"] = pd.to_datetime(frame["DateObj"], errors="coerce")
    return annotate_transfer_types(frame, getattr(engine, "account_holder_names", ()))

def filter_question_period(
    engine,
    frame: pd.DataFrame,
    question: str,
) -> tuple[pd.DataFrame, str]:
    """Apply supported relative and explicit date ranges to a chat query."""
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

def question_filters(
    engine,
    frame: pd.DataFrame,
    question: str,
) -> tuple[pd.DataFrame, str, str | None, str | None, str | None, str | None]:
    """Resolve direction, category, mode, merchant, and amount constraints."""
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

def answer_statement_question(engine, question: str) -> str:
    """Answer a natural-language question using the active statement only."""
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

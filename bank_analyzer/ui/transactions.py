from __future__ import annotations

import re
import pandas as pd
import streamlit as st

from bank_analyzer.services.analytics_service import transaction_totals
from bank_analyzer.services.transfer_service import annotate_transfer_types
from bank_analyzer.utils.exports import csv_download_bytes
from bank_analyzer.utils.formatting import format_currency

def render_transactions(engine) -> None:
    if not engine.raw_transactions:
        return

    records = pd.DataFrame(engine.raw_transactions)
    records["Date"] = pd.to_datetime(records["Date"], errors="coerce")
    for column in ("Withdrawal", "Deposit", "Amount"):
        records[column] = pd.to_numeric(records[column], errors="coerce").fillna(0.0)
    records = records.dropna(subset=["Date"])
    records = annotate_transfer_types(
        records, getattr(engine, "account_holder_names", ())
    )
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
    totals = transaction_totals(
        filtered, getattr(engine, "account_holder_names", ())
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("Matching transactions", f"{totals['transaction_count']:,}", border=True)
    summary_columns[1].metric("Money in", format_currency(totals["money_in"]), border=True)
    summary_columns[2].metric("Money out", format_currency(totals["money_out"]), border=True)
    summary_columns[3].metric("Net spending", format_currency(totals["net_spending"]), border=True)
    with st.expander("How these filtered totals are calculated"):
        st.caption(
            f"Applied filters: {start_date:%d %b %Y}–{end_date:%d %b %Y} · "
            f"{transaction_type} · {len(filtered):,} matching transaction(s) · "
            f"amount {format_currency(amount_range[0])}–{format_currency(amount_range[1])}."
        )
        reconciliation = st.columns(3)
        reconciliation[0].metric("Sent to others", format_currency(totals["sent_to_others"]))
        reconciliation[1].metric("Identified returns", format_currency(totals["refunds"]))
        reconciliation[2].metric("Own-account transfers", format_currency(totals["transfers"]))
        st.caption(
            f"Net spending = withdrawals excluding own-account transfers "
            f"({format_currency(totals['external_withdrawals'])}) − identified returns "
            f"({format_currency(totals['refunds'])}) = {format_currency(totals['net_spending'])}. "
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
        data=csv_download_bytes(filtered),
        file_name="filtered_transactions.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )

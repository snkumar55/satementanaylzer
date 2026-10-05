from __future__ import annotations

import copy
import hashlib
import io
import inspect
import sys
from dataclasses import MISSING
from pathlib import Path

import pandas as pd
import streamlit as st
from reflex_base.event import EventHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bank_analyzer.bank_analyzer import AnalyzerState
except ImportError:
    from bank_analyzer.bank_analyzer.bank_analyzer import AnalyzerState

from bank_analyzer.ui.chat import render_chat
from bank_analyzer.ui.overview import render_overview
from bank_analyzer.ui.spending_intelligence import render_spending_intelligence
from bank_analyzer.ui.transactions import render_transactions
from bank_analyzer.utils.filters import (
    EXCLUSION_WARNING_RATIO,
    apply_transaction_exclusions,
)
from bank_analyzer.utils.exports import csv_download_bytes
from bank_analyzer.utils.formatting import format_currency

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

def build_filtered_engine(engine, filtered_records):
    analysis_engine = copy.deepcopy(engine)
    source = filtered_records[["Date", "Description", "Withdrawal", "Deposit"]].copy()
    source = source.rename(
        columns={"Description": "Narration", "Withdrawal": "Debit", "Deposit": "Credit"}
    )
    analysis_engine.process_transactions(source, clean_statement_rows=False)
    return analysis_engine

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
                st.session_state.pop("all_raw_transactions", None)
                st.session_state.pop("uploaded_filename", None)
                engine.raw_transactions = []
                load_statement(engine, uploaded_file)
                st.session_state.uploaded_signature = signature
                st.session_state.uploaded_filename = uploaded_file.name
                engine.chat_messages = [["assistant", "Statement loaded. Ask a question about its transactions."]]
                st.session_state.all_raw_transactions = copy.deepcopy(engine.raw_transactions)
                st.session_state.chat_messages = copy.deepcopy(engine.chat_messages)
            except (ValueError, pd.errors.ParserError, ImportError) as error:
                st.error(str(error))
            except Exception as error:
                print(f"Statement processing failed for {uploaded_file.name}: {error}")
                st.error("We couldn't read this statement. Check its date and deposit/withdrawal columns, then try again.")

    all_records = st.session_state.get("all_raw_transactions", engine.raw_transactions)
    if not all_records:
        st.info("Your overview, transaction search, and statement Q&A will appear here after upload.")
        return

    st.success("✅ Statement Loaded")
    uploaded_records = pd.DataFrame(all_records)
    uploaded_dates = pd.to_datetime(uploaded_records["Date"], errors="coerce").dropna()
    upload_summary = st.columns(3)
    upload_summary[0].metric(
        "File name", st.session_state.get("uploaded_filename", "Uploaded statement")
    )
    upload_summary[1].metric("Transaction count", f"{len(uploaded_records):,}")
    upload_summary[2].metric(
        "Date range",
        (
            f"{uploaded_dates.min():%d %b %Y} – {uploaded_dates.max():%d %b %Y}"
            if not uploaded_dates.empty
            else "No valid transaction dates"
        ),
    )

    with st.sidebar:
        st.subheader("Exclude Transactions")
        exclusion_input = st.text_area(
            "Enter keywords",
            placeholder="AMAZON\nCREDIT CARD PAYMENT\nSELF TRANSFER",
            help="Enter one keyword per line or separate keywords with commas. Matching is case-insensitive and checks transaction descriptions.",
            key="transaction_exclusion_keywords",
            height=120,
        )
        account_holder_input = st.text_input(
            "Account holder name (optional)",
            help="Use your name only if it appears in statement descriptions for transfers to your own account. Separate multiple names with commas.",
            key="account_holder_names",
        )

    try:
        filtered_records, excluded_records = apply_transaction_exclusions(
            pd.DataFrame(all_records), exclusion_input
        )
    except ValueError as error:
        st.error(str(error))
        return

    excluded_amount = 0.0
    if not excluded_records.empty:
        movements = excluded_records[["Deposit", "Withdrawal"]].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        excluded_amount = float(movements.max(axis=1).sum())

    with st.sidebar:
        st.metric("Excluded Transactions", f"{len(excluded_records):,}")
        st.metric("Excluded Amount", format_currency(excluded_amount))
        excluded_share = len(excluded_records) / len(all_records)
        if excluded_share > EXCLUSION_WARNING_RATIO:
            st.warning(
                f"Warning: {excluded_share:.0%} of transactions are currently excluded from analysis."
            )

    if not excluded_records.empty:
        with st.expander("View Excluded Transactions"):
            audit = excluded_records[["Date", "Description", "Amount", "Category"]].copy()
            audit["Date"] = pd.to_datetime(audit["Date"], errors="coerce").dt.strftime("%d %b %Y")
            audit["Amount"] = audit["Amount"].map(format_currency)
            st.dataframe(audit, hide_index=True, width="stretch")
            st.download_button(
                "Download Excluded Transactions CSV",
                data=csv_download_bytes(excluded_records),
                file_name="excluded_transactions.csv",
                mime="text/csv",
            )

    if filtered_records.empty:
        st.info("All transactions are currently excluded. Change or clear the exclusion keywords to view analysis.")
        return

    loaded_name = st.session_state.get("uploaded_filename")
    if loaded_name:
        st.caption(f"Currently analyzing **{loaded_name}** · {len(filtered_records):,} of {len(all_records):,} transactions")

    active_engine = engine if excluded_records.empty else build_filtered_engine(engine, filtered_records)
    active_engine.account_holder_names = tuple(
        name.strip() for name in account_holder_input.split(",") if name.strip()
    )
    active_engine.statement_filename = st.session_state.get(
        "uploaded_filename", "Uploaded statement"
    )
    active_engine.chat_messages = copy.deepcopy(
        st.session_state.get("chat_messages", engine.chat_messages)
    )
    overview_tab, transactions_tab, chat_tab, intelligence_tab = st.tabs(
        ["Overview", "Transactions", "Chat", "Spending Intelligence"]
    )
    with overview_tab:
        render_overview(active_engine)
    with transactions_tab:
        render_transactions(active_engine)
    with chat_tab:
        render_chat(active_engine)
    with intelligence_tab:
        render_spending_intelligence(active_engine)

if __name__ == "__main__":
    main()

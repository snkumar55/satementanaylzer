from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bank_analyzer.services.analytics_service import CHART_COLORS, dashboard_summary
from bank_analyzer.services.forecast_service import forecast_frame
from bank_analyzer.services.merchant_service import merchant_summary
from bank_analyzer.utils.formatting import format_currency

def render_overview(engine) -> None:
    """Render the statement summary and its core analysis views."""
    records = pd.DataFrame(engine.raw_transactions)
    summary = dashboard_summary(
        records, getattr(engine, "account_holder_names", ())
    )
    records = summary["records"]
    st.subheader("Overview")
    period_start, period_end = summary["period_start"], summary["period_end"]
    period_label = (
        f"{period_start:%d %b %Y} – {period_end:%d %b %Y}"
        if period_start is not None and period_end is not None
        else "No valid transaction dates"
    )

    st.markdown("#### Statement summary")
    statement_summary = st.columns(3)
    statement_summary[0].metric("File", getattr(engine, "statement_filename", "Uploaded statement"))
    statement_summary[1].metric("Analysis period", period_label)
    statement_summary[2].metric("Transactions", f"{summary['transaction_count']:,}")

    st.markdown("#### Primary indicators")
    primary = st.columns(3)
    primary[0].metric("Money in", format_currency(summary["money_in"]), border=True)
    primary[1].metric("Money out", format_currency(summary["money_out"]), border=True)
    primary[2].metric(
        "Net spending",
        format_currency(summary["net_spending"]),
        help="External withdrawals less identified refunds; own-account transfers are excluded.",
        border=True,
    )

    st.markdown("#### Additional indicators")
    secondary = st.columns(3)
    secondary[0].metric("Refunds", format_currency(summary["refunds"]), border=True)
    secondary[1].metric("Transfers", format_currency(summary["transfers"]), border=True)
    secondary[2].metric("Transaction count", f"{summary['transaction_count']:,}", border=True)

    external_withdrawals = records[
        (records["Withdrawal"] > 0) & ~records["SelfTransfer"]
    ].copy()
    category_totals = (
        external_withdrawals.groupby("Category", dropna=False)["Withdrawal"]
        .sum()
        .sort_values(ascending=False)
    )
    merchant_totals = merchant_summary(external_withdrawals)
    insights = []
    if not category_totals.empty:
        category_total = float(category_totals.sum())
        category_share = float(category_totals.iloc[0] / category_total) if category_total else 0.0
        insights.append(
            f"Top category: {category_totals.index[0]} · {category_share:.0%} of external spending"
        )
    if not merchant_totals.empty:
        insights.append(
            f"Top merchant: {merchant_totals.index[0]} · "
            f"{format_currency(merchant_totals.iloc[0]['Total'])}"
        )
    if not external_withdrawals.empty:
        largest = external_withdrawals.loc[external_withdrawals["Withdrawal"].idxmax()]
        insights.append(
            f"Largest expense: {format_currency(largest['Withdrawal'])} · "
            f"{largest.get('Merchant') or largest.get('Description') or 'Unknown'}"
        )
    anomalies = records[records["Anomaly"].astype(str).str.contains("Outlier", case=False, na=False)]
    recurring_count = len(engine.monthly_recurring_table)
    if recurring_count:
        insights.append(f"Recurring payments: {recurring_count:,} pattern(s) detected")
    insights.append(
        f"Unusual activity: {len(anomalies):,} transaction(s) flagged for review"
        if not anomalies.empty
        else "Unusual activity: no statistical outliers flagged"
    )
    st.markdown("#### Key insights")
    insight_columns = st.columns(min(4, max(1, len(insights))))
    for index, insight in enumerate(insights):
        with insight_columns[index % len(insight_columns)]:
            st.container(border=True).markdown(insight)

    monthly = pd.DataFrame(engine.monthly_chart_data)
    if not monthly.empty:
        st.markdown("#### Cash flow trend")
        chart_data = monthly.rename(
            columns={"month": "Month", "deposit": "Money in", "withdrawal": "Money out", "savings": "Net flow"}
        )
        cash_flow = px.line(
            chart_data,
            x="Month",
            y=["Money in", "Money out", "Net flow"],
            markers=True,
            color_discrete_map={
                "Money in": CHART_COLORS["credit"],
                "Money out": CHART_COLORS["spending"],
                "Net flow": CHART_COLORS["transfer"],
            },
            labels={"value": "Amount (₹)", "variable": ""},
        )
        cash_flow.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend_title_text="",
            margin=dict(l=8, r=12, t=12, b=8),
            height=360,
        )
        cash_flow.update_traces(hovertemplate="%{x}<br>₹%{y:,.2f}<extra>%{fullData.name}</extra>")
        st.plotly_chart(cash_flow, width="stretch", config={"displayModeBar": False})

    left, right = st.columns(2)
    with left:
        st.markdown("#### Category analysis")
        if category_totals.empty:
            st.info("No external spending is available for category analysis.")
        else:
            categories = (
                category_totals.head(10)
                .rename("Total")
                .rename_axis("Category")
                .reset_index()
                .sort_values("Total")
            )
            category_chart = px.bar(
                categories,
                x="Total",
                y="Category",
                orientation="h",
                color_discrete_sequence=[CHART_COLORS["spending"]],
                labels={"Total": "Amount (₹)", "Category": ""},
            )
            category_chart.update_layout(
                template="plotly_white",
                showlegend=False,
                margin=dict(l=8, r=8, t=12, b=8),
                height=300,
            )
            category_chart.update_traces(hovertemplate="%{y}<br>₹%{x:,.2f}<extra></extra>")
            st.plotly_chart(category_chart, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("#### Merchant analysis")
        if merchant_totals.empty:
            st.info("No merchants are available for analysis.")
        else:
            merchant_chart = px.bar(
                merchant_totals.head(10).reset_index().sort_values("Total"),
                x="Total",
                y="Merchant",
                orientation="h",
                color_discrete_sequence=[CHART_COLORS["insight"]],
                labels={"Total": "Amount (₹)", "Merchant": ""},
            )
            merchant_chart.update_layout(
                template="plotly_white",
                showlegend=False,
                margin=dict(l=8, r=8, t=12, b=8),
                height=300,
            )
            merchant_chart.update_traces(hovertemplate="%{y}<br>₹%{x:,.2f}<extra></extra>")
            st.plotly_chart(merchant_chart, width="stretch", config={"displayModeBar": False})
            merchant_display = merchant_totals.head(10).reset_index()
            merchant_display["Total spent"] = merchant_display["Total"].map(format_currency)
            merchant_display["Average transaction"] = merchant_display["Average"].map(format_currency)
            st.dataframe(
                merchant_display[
                    ["Merchant", "Transactions", "Total spent", "Average transaction"]
                ].rename(columns={"Transactions": "Count"}),
                hide_index=True,
                width="stretch",
            )

    payment_methods = (
        external_withdrawals.groupby("PaymentMode", dropna=False)["Withdrawal"]
        .sum()
        .sort_values(ascending=False)
    )
    if not payment_methods.empty:
        st.markdown("#### Payment method analysis")
        method_chart = px.bar(
            payment_methods.rename("Total").rename_axis("Payment method").reset_index(),
            x="Payment method",
            y="Total",
            color_discrete_sequence=[CHART_COLORS["transfer"]],
            labels={"Total": "Amount (₹)"},
        )
        method_chart.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=8, r=8, t=12, b=8),
            height=300,
        )
        method_chart.update_traces(hovertemplate="%{x}<br>₹%{y:,.2f}<extra></extra>")
        st.plotly_chart(method_chart, width="stretch", config={"displayModeBar": False})

    sources = (
        records[(records["Deposit"] > 0) & ~records["SelfTransfer"]]
        .groupby("Merchant", dropna=False)
        .agg(Transactions=("Deposit", "size"), Total=("Deposit", "sum"))
        .sort_values("Total", ascending=False)
        .head(10)
    )
    if not sources.empty:
        st.markdown("#### Deposit sources")
        source_display = sources.reset_index()
        source_display["Total credited"] = source_display["Total"].map(format_currency)
        st.dataframe(
            source_display[["Merchant", "Transactions", "Total credited"]].rename(
                columns={"Merchant": "Source"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Recurring payments")
    recurring = pd.DataFrame(
        engine.monthly_recurring_table,
        columns=[
            "Merchant", "Category", "Mode", "Months paid", "Average per month",
            "Total paid", "Monthly range", "Latest month",
        ],
    )
    if recurring.empty:
        st.info("No recurring payments were identified in this statement.")
    else:
        st.caption("Payments to the same merchant found across multiple statement months.")
        st.dataframe(recurring, hide_index=True, width="stretch")

    st.markdown("#### Unusual transactions")
    if anomalies.empty:
        st.info("No statistical outliers were flagged. Unusual amounts are not necessarily errors.")
    else:
        anomaly_display = anomalies.copy()
        anomaly_display["Date"] = pd.to_datetime(anomaly_display["Date"], errors="coerce").dt.strftime("%d %b %Y")
        for column in ("Withdrawal", "Deposit"):
            anomaly_display[column] = anomaly_display[column].map(format_currency)
        st.dataframe(
            anomaly_display[["Date", "Description", "Category", "Withdrawal", "Deposit"]],
            hide_index=True,
            width="stretch",
        )

    if engine.forecast_table:
        st.markdown("#### Next-month estimate")
        st.caption("A trend-based estimate from historical monthly totals, not a guarantee.")
        forecast = forecast_frame(engine)
        for column in ("Deposit", "Withdrawal", "Savings"):
            forecast[column] = forecast[column].apply(
                lambda value: value if value == "N/A" else format_currency(
                    float(str(value).replace("₹", "").replace(",", ""))
                )
            )
        st.dataframe(forecast, hide_index=True, width="stretch")

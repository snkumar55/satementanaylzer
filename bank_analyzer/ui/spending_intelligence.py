from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bank_analyzer.services.analytics_service import (
    CHART_COLORS,
    spending_health_score,
    spending_intelligence_data,
)
from bank_analyzer.services.merchant_service import merchant_summary
from bank_analyzer.utils.formatting import format_currency

def render_spending_intelligence(engine) -> None:
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

    merchant_totals = merchant_summary(spending)
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
            color_discrete_sequence=[CHART_COLORS["spending"]],
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
        color_discrete_sequence=[CHART_COLORS["spending"]],
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
    weekday_rows = spending.loc[spending["Date"].dt.dayofweek < 5]
    weekend_rows = spending.loc[spending["Date"].dt.dayofweek >= 5]
    weekday_spend = float(weekday_rows["Withdrawal"].sum())
    weekend_spend = float(weekend_rows["Withdrawal"].sum())
    weekday_days = max(1, spending.loc[spending["Date"].dt.dayofweek < 5, "Date"].dt.date.nunique())
    weekend_days = max(1, spending.loc[spending["Date"].dt.dayofweek >= 5, "Date"].dt.date.nunique())
    weekday_daily_average = weekday_spend / weekday_days
    weekend_daily_average = weekend_spend / weekend_days
    weekend_change = (
        (weekend_daily_average / weekday_daily_average - 1) * 100
        if weekday_daily_average > 0
        else None
    )
    active_days = max(1, (spending["Date"].max().date() - spending["Date"].min().date()).days + 1)
    daily_average = total_spending / active_days
    behaviour = [
        ("Largest category", f"{lead_category} · {lead_share:.0%} of spending"),
        ("Fastest-growing category", "Not enough month history" if len(monthly) < 2 else "See category trend below"),
        ("Most frequent merchant", f"{merchant_totals.sort_values(['Transactions', 'Total'], ascending=False).index[0]} · {int(most_frequent['Transactions'])} transactions"),
        ("Largest purchase", f"{format_currency(largest_expense['Withdrawal'])} · {largest_expense['Merchant']}"),
        (
            "Weekday vs weekend",
            f"{format_currency(weekday_spend)} weekdays · {format_currency(weekend_spend)} weekends"
            + (
                f" · weekends {abs(weekend_change):.0f}% "
                f"{'higher' if weekend_change >= 0 else 'lower'} per active day"
                if weekend_change is not None
                else ""
            ),
        ),
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
    category_merchants = merchant_summary(category_spend, limit=10)
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
                color_discrete_sequence=[CHART_COLORS["spending"]],
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
        color_discrete_sequence=[CHART_COLORS["insight"]],
    )
    score_chart.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=8, b=8), height=240)
    st.plotly_chart(score_chart, width="stretch", config={"displayModeBar": False})
    st.markdown("**Score drivers**")
    for driver in score_drivers:
        st.caption(f"• {driver}")

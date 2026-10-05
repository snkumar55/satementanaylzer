# Changelog

## Unreleased

- Split Streamlit renderers, analysis helpers, filters, exports, and formatting into `ui/`, `services/`, and `utils/` modules.
- Moved keyword-based categorization and forecast calculations into reusable services while keeping the existing analyzer pipeline and controls intact.
- Centralized overview and transaction KPI calculations and merchant summaries.
- Redesigned the Overview around statement metadata, cash-flow KPIs, insights, Plotly trends, category/merchant analysis, recurring payments, unusual activity, and forecasts.
- Added Plotly visualizations to the Overview while retaining existing Spending Intelligence charts and transaction/chat workflows.
- Kept transaction exclusions applied consistently to all analysis and exposed optional account-holder matching instead of a hardcoded personal name.
- Added CSV export utility, architecture/migration documentation, and this release log.

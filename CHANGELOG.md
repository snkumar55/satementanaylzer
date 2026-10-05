# Changelog

## Unreleased

- Split Streamlit renderers, analysis helpers, filters, exports, and formatting into `ui/`, `services/`, and `utils/` modules.
- Added reusable categorization and forecast services while keeping the existing analyzer pipeline and controls intact; the Reflex state loads them lazily to avoid Streamlit import-time registration errors.
- Centralized overview and transaction KPI calculations and merchant summaries.
- Redesigned the Overview around statement metadata, cash-flow KPIs, insights, Plotly trends, category/merchant analysis, recurring payments, unusual activity, and forecasts.
- Added Plotly visualizations to the Overview while retaining existing Spending Intelligence charts and transaction/chat workflows.
- Kept transaction exclusions applied consistently to all analysis and exposed optional account-holder matching instead of a hardcoded personal name.
- Replaced the Streamlit analyzer's Reflex-backed adapter with a standalone pandas/sklearn engine; the legacy Reflex application remains separate.
- Added CSV export utility, architecture/migration documentation, and this release log.

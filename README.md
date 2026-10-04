# Bank Statement Analyzer

A local web app for analyzing bank statement transactions, viewing summaries and charts, filtering transactions, and asking questions about the loaded statement.

## Run locally

Install dependencies and launch Streamlit from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run bank_analyzer/streamlit_app.py
```

Streamlit opens the app in your browser, usually at **http://localhost:8501**. Leave the terminal open while using the app. Press **Ctrl+C** to stop it.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run bank_analyzer/streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep the root `requirements.txt` and the `bank_analyzer/` directory in the repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and select **Create app**.
3. Select the repository and branch, and set **Main file path** to `bank_analyzer/streamlit_app.py`.
4. Select **Deploy**. Streamlit Cloud installs the dependencies from the root `requirements.txt` and provides a public app URL.

This workspace is not connected to a GitHub repository, so the app is prepared for deployment but has not been published to Streamlit Community Cloud.

## Use the analyzer

Upload a CSV, XLSX, or XLS bank statement and wait for its summaries and charts to load. Use the transaction filters to narrow the displayed data. Open **Chat** to ask questions about the loaded statement, for example:

- `What is the highest amount credited?`
- `What is the highest amount withdrawn?`
- `How much did I spend at Swiggy last month?`
- `Show my monthly totals for the last 3 months`
- `Which category had the highest spending?`
- `Show repeated payments`

The **Transactions** tab keeps filtering simple: search keywords across descriptions, payees, categories, and payment methods; choose a date range and transaction type; optionally expand the amount range filter. Transaction types include deposits, withdrawals, payments sent to others, identified returns, and own-account transfers. Separate search words must all match. The displayed deposit, gross withdrawal, net spending, returned amount, sent-to-others, and own-account transfer totals are calculated only from matching rows, and the spending formula is shown above the results. Identified returns reduce net spending; transfers mentioning the account holder (`Sharath`) are listed separately and excluded from spending. Download the filtered rows as CSV.

Use **Exclude Transactions** in the sidebar to remove one or more narration/description keywords from every analysis, including dashboard totals, charts, forecasts, chat, and Spending Intelligence. Keywords are case-insensitive, support partial matches, and can be entered on separate lines or comma-separated. The sidebar shows excluded counts and amounts; the excluded rows can be reviewed and downloaded as CSV.

The **Overview** emphasizes net spending, with income and gross withdrawals alongside it. Reconciliation details are available under **How totals are calculated**. It also includes monthly cash flow, deposit sources, payment methods, recurring payments, category breakdowns, transactions to review, and a trend-based estimate. **Spending Intelligence** summarizes category and merchant spending, concentration, monthly trends, largest purchases, spending behaviour, category details, and a statement-based score. Return and own-account transfer labels are inferred from statement descriptions. Chat questions can ask for totals, highest/lowest transactions, averages, counts, merchant and category breakdowns, payment methods, recurring payments, monthly summaries, and date/year ranges. Chat responses are calculated from the statement loaded for that session.

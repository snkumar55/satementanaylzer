from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def compute_forecast(
    monthly_df: pd.DataFrame,
    forecast_model: str,
    trend_parameter: str,
    smoothing_level: float,
) -> list[list[str]]:
    """Forecast the next monthly deposits, withdrawals, and net cash flow."""
    try:
        if len(monthly_df) < 2:
            return [["Next Month (forecast)", "N/A", "N/A", "N/A"]]
        monthly_numeric = monthly_df.reset_index(drop=True)
        if forecast_model == "linear":
            monthly_numeric["idx"] = np.arange(len(monthly_numeric))
            features = monthly_numeric[["idx"]].values
            deposit_model = LinearRegression().fit(
                features, monthly_numeric["Deposit"].values
            )
            withdrawal_model = LinearRegression().fit(
                features, monthly_numeric["Withdrawal"].values
            )
            next_index = np.array([[len(monthly_numeric)]])
            predicted_deposit = max(0.0, deposit_model.predict(next_index)[0])
            predicted_withdrawal = max(0.0, withdrawal_model.predict(next_index)[0])
        else:
            trend = (
                None
                if trend_parameter in ("None", "none", "")
                else trend_parameter
            )
            try:
                deposit_series = monthly_numeric["Deposit"].astype(float).values
                deposit_model = ExponentialSmoothing(
                    deposit_series,
                    trend=trend,
                    seasonal=None,
                    initialization_method="estimated",
                )
                deposit_fit = deposit_model.fit(
                    smoothing_level=smoothing_level,
                    optimized=True,
                )
                predicted_deposit = max(0.0, float(deposit_fit.forecast(1)[0]))
            except Exception:
                predicted_deposit = float(monthly_numeric["Deposit"].iloc[-1])
            try:
                withdrawal_series = monthly_numeric["Withdrawal"].astype(float).values
                withdrawal_model = ExponentialSmoothing(
                    withdrawal_series,
                    trend=trend,
                    seasonal=None,
                    initialization_method="estimated",
                )
                withdrawal_fit = withdrawal_model.fit(
                    smoothing_level=smoothing_level,
                    optimized=True,
                )
                predicted_withdrawal = max(
                    0.0, float(withdrawal_fit.forecast(1)[0])
                )
            except Exception:
                predicted_withdrawal = float(monthly_numeric["Withdrawal"].iloc[-1])
        predicted_savings = predicted_deposit - predicted_withdrawal
        return [[
            "Next Month (forecast)",
            f"₹{predicted_deposit:,.2f}",
            f"₹{predicted_withdrawal:,.2f}",
            f"₹{predicted_savings:,.2f}",
        ]]
    except Exception:
        return [["Next Month (forecast)", "N/A", "N/A", "N/A"]]


def forecast_frame(engine) -> pd.DataFrame:
    """Return the existing analyzer forecast in a consistently named frame."""
    return pd.DataFrame(
        engine.forecast_table,
        columns=["Period", "Deposit", "Withdrawal", "Savings"],
    )

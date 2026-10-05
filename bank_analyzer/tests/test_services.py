from __future__ import annotations

import unittest

import pandas as pd

from bank_analyzer.services.analytics_service import dashboard_summary, transaction_totals
from bank_analyzer.services.categorization_service import categorize_description
from bank_analyzer.services.forecast_service import compute_forecast
from bank_analyzer.services.transfer_service import annotate_transfer_types
from bank_analyzer.utils.exports import csv_download_bytes
from bank_analyzer.utils.filters import apply_transaction_exclusions

class TransactionExclusionTests(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame(
            {
                "Narration": [
                    "AMAZON PAY",
                    "Amazon Seller Services",
                    "CREDIT CARD PAYMENT",
                    "Salary",
                ],
                "Withdrawal": [10.0, 20.0, 30.0, 0.0],
                "Deposit": [0.0, 0.0, 0.0, 100.0],
            }
        )

    def test_keywords_match_case_insensitive_partial_descriptions(self):
        filtered, excluded = apply_transaction_exclusions(
            self.records, "amazon,\ncredit card"
        )
        self.assertEqual(filtered["Narration"].tolist(), ["Salary"])
        self.assertEqual(len(excluded), 3)

    def test_empty_keywords_preserve_all_rows(self):
        filtered, excluded = apply_transaction_exclusions(self.records, "")
        pd.testing.assert_frame_equal(filtered, self.records)
        self.assertTrue(excluded.empty)

class TransactionAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame(
            {
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "Description": ["UPI TO TAYLOR", "SELF TRANSFER", "GROCERY STORE"],
                "Merchant": ["Transfer", "Transfer", "Market"],
                "Category": ["Other", "Other", "Groceries"],
                "PaymentMode": ["UPI", "UPI", "Card"],
                "Withdrawal": [50.0, 75.0, 25.0],
                "Deposit": [0.0, 0.0, 0.0],
                "Amount": [-50.0, -75.0, -25.0],
            }
        )

    def test_owner_name_is_configurable_and_not_assumed(self):
        without_name = annotate_transfer_types(self.records)
        with_name = annotate_transfer_types(self.records, ["Taylor"])
        self.assertFalse(without_name.loc[0, "SelfTransfer"])
        self.assertTrue(with_name.loc[0, "SelfTransfer"])
        self.assertTrue(with_name.loc[1, "SelfTransfer"])

    def test_dashboard_and_transaction_kpis_share_transfer_rules(self):
        summary = dashboard_summary(self.records, ["Taylor"])
        totals = transaction_totals(self.records, ["Taylor"])
        self.assertEqual(summary["transaction_count"], 3)
        self.assertEqual(summary["transfers"], 125.0)
        self.assertEqual(summary["net_spending"], 25.0)
        self.assertEqual(totals["net_spending"], 25.0)

    def test_csv_export_is_utf8(self):
        result = csv_download_bytes(self.records)
        self.assertTrue(result.startswith(b"Date,Description"))
        self.assertIn(b"GROCERY STORE", result)

    def test_categorization_uses_existing_ordered_rules(self):
        rules = {"Food": ["cafe", "restaurant"], "Bills": ["utility"]}
        self.assertEqual(categorize_description("LOCAL CAFE", rules), "Food")
        self.assertEqual(categorize_description("POWER UTILITY", rules), "Bills")
        self.assertEqual(categorize_description("Unmatched", rules), "Other")

    def test_linear_forecast_preserves_monthly_projection(self):
        monthly = pd.DataFrame(
            {"Deposit": [100.0, 200.0], "Withdrawal": [50.0, 75.0]}
        )
        forecast = compute_forecast(monthly, "linear", "add", 0.2)
        self.assertEqual(forecast[0][0], "Next Month (forecast)")
        self.assertEqual(forecast[0][1:], ["₹300.00", "₹100.00", "₹200.00"])

if __name__ == "__main__":
    unittest.main()

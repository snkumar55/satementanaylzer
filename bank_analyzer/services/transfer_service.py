from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

def annotate_transfer_types(
    records: pd.DataFrame,
    account_holder_names: Iterable[str] = (),
) -> pd.DataFrame:
    """Add inferred transfer, return, and transaction-type labels."""
    annotated = records.copy()
    for column in ("Description", "Merchant", "PaymentAccountDisplay", "PaymentMode", "Category"):
        if column not in annotated:
            annotated[column] = ""
        annotated[column] = annotated[column].fillna("").astype(str)

    narration = (
        annotated["Description"] + " " + annotated["Merchant"] + " "
        + annotated["PaymentAccountDisplay"]
    ).str.casefold()
    own_account_pattern = (
        r"\bself(?:\s|-)?transfer\b|\bown\s+account\b|"
        r"\bbetween\s+(?:my|own)\s+accounts?\b|\btransfer(?:red)?\s+to\s+myself\b"
    )
    if isinstance(account_holder_names, str):
        account_holder_names = [account_holder_names]
    names = [
        re.escape(name.strip())
        for name in account_holder_names
        if isinstance(name, str) and name.strip()
    ]
    if names:
        own_account_pattern += rf"|\b(?:{'|'.join(names).casefold()})\b"
    annotated["SelfTransfer"] = narration.str.contains(
        own_account_pattern,
        case=False,
        regex=True,
        na=False,
    )
    annotated["ReturnedPayment"] = (
        (annotated["Deposit"] > 0)
        & ~annotated["SelfTransfer"]
        & narration.str.contains(r"\b(?:refund|refunded|return(?:ed)?|reversal|reversed|chargeback|re-credited)\b", regex=True, na=False)
    )
    transfer_signals = narration.str.contains(
        r"\b(?:transfer|transferred|sent|imps|neft|rtgs|p2p|p2a|beneficiary)\b",
        regex=True,
        na=False,
    )
    recipient_signal = narration.str.contains(r"\bto\s+[a-z][a-z0-9._-]*\b", regex=True, na=False)
    annotated["SentToOthers"] = (
        (annotated["Withdrawal"] > 0)
        & ~annotated["SelfTransfer"]
        & annotated["PaymentMode"].str.contains(r"UPI|Transfer", case=False, regex=True)
        & (transfer_signals | (recipient_signal & annotated["Category"].eq("Other")))
    )
    merchant_key = (
        annotated["Merchant"]
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    for index, row in annotated[annotated["Deposit"] > 0].iterrows():
        if row["SelfTransfer"] or annotated.at[index, "ReturnedPayment"]:
            continue
        prior_sent = annotated[
            annotated["SentToOthers"]
            & (annotated["Date"] < row["Date"])
            & ((annotated["Withdrawal"] - row["Deposit"]).abs() < 0.01)
            & merchant_key.eq(merchant_key.at[index])
            & merchant_key.ne("")
            & merchant_key.ne("unknown")
        ]
        if not prior_sent.empty:
            annotated.at[index, "ReturnedPayment"] = True
    annotated["TransactionType"] = "Payment"
    annotated.loc[annotated["Deposit"] > 0, "TransactionType"] = "Deposit"
    annotated.loc[annotated["SentToOthers"], "TransactionType"] = "Sent to someone"
    annotated.loc[annotated["ReturnedPayment"], "TransactionType"] = "Money returned"
    annotated.loc[annotated["SelfTransfer"], "TransactionType"] = "Own-account transfer"
    annotated["ReturnedAmount"] = annotated["Deposit"].where(annotated["ReturnedPayment"], 0.0)
    annotated["SelfTransferAmount"] = annotated[["Deposit", "Withdrawal"]].max(axis=1).where(
        annotated["SelfTransfer"], 0.0
    )
    return annotated

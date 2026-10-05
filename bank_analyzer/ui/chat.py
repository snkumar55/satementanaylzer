from __future__ import annotations

import streamlit as st

from bank_analyzer.services.chat_service import answer_statement_question

def render_chat(engine) -> None:
    st.subheader("Ask about this statement")
    for speaker, message in engine.chat_messages:
        with st.chat_message("assistant" if speaker == "assistant" else "user"):
            st.markdown(message)

    with st.form("payment_chat_form", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="What is the highest amount credited?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        messages = list(engine.chat_messages)
        messages.extend([
            ["user", question.strip()],
            ["assistant", answer_statement_question(engine, question.strip())],
        ])
        engine.chat_messages = messages[-18:]
        st.session_state.chat_messages = engine.chat_messages
        st.rerun()

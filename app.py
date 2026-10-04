"""Streamlit UI: chat, visible agent trace, price charts, and a sidebar for reports, portfolio and memory.

Run with:  streamlit run app.py
"""
import json
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors

load_dotenv()

import agent  # noqa: E402  (reads AGENT_MODEL from the environment loaded above)
import ingest  # noqa: E402
import rag  # noqa: E402
from tools import Context  # noqa: E402

st.set_page_config(page_title="Finance Research Agent", page_icon="📈", layout="wide")

# Fixed series colours, assigned in this order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
EXAMPLES = [
    "Should I look into Tata Motors?",
    "Compare TCS and Infosys over the last 2 years",
    "What risks does the company mention in its annual report?",
    "How diversified is my portfolio?",
]

state = st.session_state
state.setdefault("messages", [])   # full API conversation = the agent's short-term memory
state.setdefault("history", [])    # what the chat window shows
state.setdefault("ctx", Context())
ctx = state.ctx


def price_chart(charts):
    """One line per ticker. Several tickers are rebased to 100 so they share one axis."""
    compare = len(charts) > 1
    fig = go.Figure()
    for i, c in enumerate(charts):
        y = [v / c["close"][0] * 100 for v in c["close"]] if compare else c["close"]
        fig.add_trace(go.Scatter(
            x=c["dates"], y=y, name=c["ticker"], mode="lines",
            line=dict(width=2, color=SERIES_COLORS[i % len(SERIES_COLORS)]),
            hovertemplate="%{y:,.2f}",
        ))
    if compare:
        title, y_title = "Price performance, rebased to 100", "Start of period = 100"
    else:
        title, y_title = f"{charts[0]['ticker']} closing price", charts[0].get("currency") or "Price"
    fig.update_layout(
        title=title, yaxis_title=y_title, hovermode="x unified", showlegend=compare,
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


def show_trace(trace):
    for step in trace:
        if step["type"] == "tool_call":
            st.markdown(f"**→ {step['name']}** `{json.dumps(step['input'])}`")
        else:
            icon = "⚠️" if step["is_error"] else "←"
            preview = step["output"] if len(step["output"]) <= 600 else step["output"][:600] + " …"
            st.markdown(f"{icon} result")
            st.code(preview, language="json")


def show_turn(turn):
    with st.chat_message(turn["role"]):
        if turn.get("trace"):
            with st.expander(f"Agent trace: {sum(s['type'] == 'tool_call' for s in turn['trace'])} tool calls"):
                show_trace(turn["trace"])
        st.markdown(turn["text"])
        if turn.get("charts"):
            st.plotly_chart(price_chart(turn["charts"]), width="stretch", key=f"chart_{turn['id']}")


# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")
    ctx.user_id = st.text_input("Username", value=ctx.user_id, help="Memories are stored per username.").strip() or "guest"

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password", help="Or set GEMINI_API_KEY in .env")

    st.divider()
    st.subheader("Portfolio")
    csv = st.file_uploader("Holdings CSV (ticker, quantity, buy_price)", type="csv")
    if csv:
        try:
            df = pd.read_csv(csv)
            df.columns = [c.strip().lower() for c in df.columns]
            ctx.portfolio = df[["ticker", "quantity", "buy_price"]]
            st.dataframe(ctx.portfolio, hide_index=True)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
    else:
        ctx.portfolio = None

    st.divider()
    st.subheader("Annual reports")
    reports = rag.list_reports()
    st.caption("Indexed: " + (", ".join(f"{c} {y}" for c, y in reports) or "none yet"))
    with st.expander("Add a report (PDF)"):
        pdf = st.file_uploader("Annual report PDF", type="pdf")
        company = st.text_input("Company key", placeholder="TCS")
        year = st.number_input("Year", min_value=1990, max_value=2100, value=2025, step=1)
        if st.button("Index report", disabled=not (pdf and company)):
            bar = st.progress(0.0, text="Embedding…")
            try:
                n = rag.add_report(
                    company, year, rag.pdf_pages(pdf.getvalue()), source=pdf.name,
                    progress=lambda done, total: bar.progress(done / total, text=f"Embedding {done}/{total} chunks"),
                )
                ctx.cache.clear()
                st.success(f"Stored {n} chunks for {company.upper()} {year}.")
            except Exception as e:
                st.error(str(e))
    with st.expander("Download a report automatically"):
        market = st.radio("Market", ["India (NSE)", "US (SEC EDGAR)"], horizontal=True)
        symbol = st.text_input("Symbol", placeholder="TCS" if market.startswith("India") else "AAPL")
        fetch = None  # set to a function(progress) -> (company, year, chunks) when the button is clicked
        if market.startswith("India"):
            if st.button("Find reports", disabled=not symbol):
                try:
                    state.nse_found = (symbol, ingest.nse_reports(symbol))
                except Exception as e:
                    state.nse_found = None
                    st.error(str(e))
            found = state.get("nse_found")
            if found and found[0] == symbol:
                choice = st.selectbox("Report", found[1], format_func=lambda r: f"{r['label']} (saved as {r['year']})")
                if st.button("Download and index"):
                    fetch = lambda progress: ingest.ingest_nse(symbol, choice["year"], progress)  # noqa: E731
        else:
            identity = os.getenv("EDGAR_IDENTITY") or st.text_input(
                "Your name and email", placeholder="Your Name your@email.com", help="SEC asks who is downloading.")
            if st.button("Download latest 10-K and index", disabled=not (symbol and identity)):
                fetch = lambda progress: ingest.ingest_edgar(symbol, identity, progress)  # noqa: E731
        if fetch:
            bar = st.progress(0.0, text="Downloading…")
            try:
                done_company, done_year, n = fetch(
                    lambda done, total: bar.progress(done / total, text=f"Embedding {done}/{total} chunks"))
                ctx.cache.clear()
                st.success(f"Stored {n} chunks for {done_company} {done_year}.")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Memory")
    memories = rag.list_memories(ctx.user_id)
    if not memories:
        st.caption("Nothing remembered about this user yet.")
    for m in memories:
        left, right = st.columns([5, 1])
        left.caption(m["fact"])
        if right.button("✕", key=f"del_{m['id']}", help="Forget this"):
            rag.delete_memory(m["id"], ctx.user_id)
            st.rerun()

    st.divider()
    if st.button("Clear chat"):
        state.messages, state.history = [], []
        ctx.cache.clear()
        st.rerun()


# ---------- main chat ----------

st.title("📈 Finance Research Agent")
st.caption("An agent that plans its own research using live market data, annual reports and what it remembers about you. "
           "For education only, not investment advice.")

for turn in state.history:
    show_turn(turn)

question = st.chat_input("Ask about a stock, a company's annual report, or your portfolio")
if not state.history and not question:
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, width="stretch"):
            question = example

if question:
    if not api_key:
        st.error("Add your Gemini API key in the sidebar, or put GEMINI_API_KEY in a .env file.")
        st.stop()

    user_turn = {"id": len(state.history), "role": "user", "text": question}
    state.history.append(user_turn)
    show_turn(user_turn)

    ctx.charts = []
    trace, answer = [], None
    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=True) as status:
            try:
                for event in agent.run(agent.make_client(api_key), state.messages, question, ctx):
                    if event["type"] == "answer":
                        answer = event["text"]
                    else:
                        trace.append(event)
                        show_trace([event])
                        if event["type"] == "tool_call":
                            status.update(label=f"Calling {event['name']}…")
                calls = sum(s["type"] == "tool_call" for s in trace)
                status.update(label=f"Done: {calls} tool calls", state="complete", expanded=False)
            except genai_errors.APIError as e:
                # Gemini reports a bad key as 400 "API key not valid", not as 401.
                if e.code in (401, 403) or "API key" in (e.message or ""):
                    answer = "The Gemini API key was rejected. Check the key in the sidebar or your .env file."
                elif e.code == 429:
                    answer = "Rate limited by the Gemini API, or the quota is used up. Wait a minute and try again."
                else:
                    answer = f"The Gemini API returned an error ({e.code}): {e.message}"
            except httpx.TransportError:
                answer = "Could not reach the Gemini API. Check your internet connection."
            if answer is None or not trace:
                status.update(label="Done", state="complete", expanded=False)

    state.history.append({
        "id": len(state.history), "role": "assistant", "text": answer, "trace": trace, "charts": ctx.charts,
    })
    st.rerun()

# Finance Research Agent

An agentic AI assistant for stock research. You ask a question such as "Should I look into Tata Motors?" and the agent decides by itself which tools to call (prices, fundamentals, news, annual reports, your portfolio), then writes a research answer with a chart. Every tool call is shown in the UI.

Built with plain function calling on the Gemini API, with no agent framework, so every part of the loop is visible in about 100 lines (`agent.py`).

> For education only. Not investment advice.

## What it does

| Feature | How |
|---|---|
| Agentic tool use | Gemini picks from 6 tools and can call several in one step |
| Live market data | `yfinance`: prices, returns, volatility, drawdown, ratios, statements (works for `.NS` stocks) |
| News | Google News RSS |
| Agentic RAG over annual reports | ChromaDB. Retrieval is a tool, so the agent searches only when the question needs it. Chunks are tagged with company, year and page, and the agent fills in those filters itself |
| Long-term memory | A second ChromaDB collection of short facts about the user. The agent saves them with a `save_memory` tool; the most relevant ones are retrieved for each question, filtered by user |
| Short-term memory | The full conversation is sent on every turn, so "they" resolves to the company you were discussing |
| Portfolio analysis | Upload a CSV of holdings; the agent values it and comments on concentration and risk |
| Visible reasoning trace | Each tool call and its result is shown in the chat |

## Architecture

```
User question
   |
   v
Memory RAG: search user_memory (filter: user_id) -> top facts -> added to the user turn
   |
   v
Gemini + conversation history (short-term memory)
   |
   |  decides which tools to call, loops until it has enough
   |
   +-- get_stock_data        live prices and risk statistics, draws the chart
   +-- get_financials        ratios and financial statements
   +-- get_news              recent headlines
   +-- search_annual_report  knowledge RAG (filter: company, year), with a cache; downloads a missing report
   +-- save_memory           stores a new fact about the user, can replace an outdated one
   +-- get_portfolio         values the uploaded holdings
   |
   v
Final answer + chart + trace
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `agent.py` | System prompt and the agent loop |
| `tools.py` | Tool schemas and their Python implementations |
| `rag.py` | ChromaDB: annual report knowledge base and user memory |
| `ingest.py` | Downloads annual reports (NSE, SEC EDGAR) and indexes them; also a command line script |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your Gemini API key in .env
streamlit run app.py
```

You can also paste the API key into the sidebar instead of using `.env`.

## Adding annual reports

You do not have to add anything. When a question needs a report that is not indexed yet, `search_annual_report` downloads and indexes it by itself: from NSE for Indian companies, and from SEC EDGAR for US ones (set `EDGAR_IDENTITY` in `.env` for those). The first search for a company takes a minute or two; after that the report is in ChromaDB and searches are instant.

NSE has no official API. The app uses the JSON endpoint behind NSE's own annual reports page, which sometimes refuses automated requests. If that happens, add the report yourself in one of these ways.

**Sidebar:** "Download a report automatically" lets you pick the year, and "Add a report (PDF)" takes a PDF you downloaded.

**Command line:**

```bash
python ingest.py --nse TCS --year 2025   # NSE; year the financial year ends in, omit for the latest
python ingest.py --edgar AAPL            # latest 10-K from SEC EDGAR
python ingest.py                         # every data/reports/COMPANY_YEAR.pdf, e.g. TCS_2025.pdf
```

## Things to try

- `Should I look into Tata Motors?` runs a full research pass with several tools
- `Compare TCS and Infosys over the last 2 years` draws both on one rebased chart
- `What risks does Apple mention in its annual report?` searches the knowledge base
- `I own 50 TCS shares bought at 3500 and I'm a long-term, low-risk investor` is saved to memory. Clear the chat, ask `How is TCS doing?`, and the answer uses it
- Upload `sample_portfolio.csv` and ask `How diversified is my portfolio?`

## Design decisions

- **Retrieval is a tool, not a fixed step.** Normal RAG searches on every message. Here "What is the TCS share price?" never touches the vector database.
- **Metadata filters.** Without the company filter, a question about TCS risks could return Infosys chunks.
- **Memories are short facts, not chat logs.** Only the facts relevant to the current question are retrieved, so the prompt stays small after many conversations.
- **Conflicting memories.** The agent sees the id of each recalled memory and passes `replaces_id` when a new fact updates an old one.
- **Stable system prompt.** The date, memories and report list go into the user turn, so the system prompt and tool definitions stay identical between requests and can be prompt-cached.
- **Tool errors go back to the model.** A wrong ticker returns an error message as the tool result, and the agent corrects itself.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Required. Get one at https://aistudio.google.com/apikey |
| `AGENT_MODEL` | `gemini-3.8-flash` | Gemini model |
| `AGENT_EFFORT` | model default | Thinking level: `minimal`, `low`, `medium`, `high` |
| `EDGAR_IDENTITY` | none | "Name email", needed only for US company reports |
# LLM-Financial-Research-agent

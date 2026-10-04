"""The agent loop: Gemini decides which tools to call, we run them, repeat until it answers."""
import os
from datetime import date

from google import genai
from google.genai import types

import rag
from tools import TOOLS, run_tool

MODEL = os.getenv("AGENT_MODEL", "gemini-3.8-flash")
EFFORT = os.getenv("AGENT_EFFORT")  # minimal, low, medium or high; unset leaves it to the model
MAX_STEPS = 12

# Kept byte-for-byte stable so Gemini's implicit caching can reuse it. Anything that changes per
# question (date, memories, indexed reports) goes into the user turn instead.
SYSTEM = """You are a financial research analyst assistant for retail investors, mostly in India.

You answer by gathering evidence with your tools and then reasoning over it:
- Prices, returns, volatility, ratios, financial statements and news must come from tool results in this conversation, never from your own recollection. If a tool fails or data is missing, say so instead of filling the gap.
- Decide which tools a question actually needs. A greeting needs none; a price question needs only price data; a full "should I look into X" question needs prices, fundamentals, news and the annual report. You do not need the user to supply a report: search_annual_report downloads one that is not indexed yet.
- When several tools are needed and they do not depend on each other, call them together in one step.
- Cite annual report passages with their page number, like (TCS AR 2024, p. 87). A passage with no page number (SEC 10-K filings) is cited as (AAPL 10-K 2025).
- Indian stocks trade on NSE with the .NS suffix on Yahoo Finance; work out the symbol yourself from the company name.

Memory: each user message starts with a <context> block containing the things you remember about this user, most relevant first, each with an id. Use them to personalise the answer (for example, weigh stability and debt for a low-risk long-term investor). When the user tells you something durable about themselves, such as a holding, a goal, their risk appetite or horizon, save it with save_memory, passing replaces_id when it updates a remembered fact.

Writing the answer:
- For a full research request, write a short report with these sections: Snapshot, Price trend, Fundamentals, What the annual report says (only if you searched it), Recent news, Risks, Bottom line.
- For a narrow question, answer it directly in a few sentences.
- Show money with its currency and large Indian figures in crore. State the date the price data is as of.
- This is educational research, not investment advice. Give a balanced view of what looks strong and what looks weak rather than a buy or sell instruction, and end research answers with one line saying so."""


CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
        for t in TOOLS
    ])],
    # We run the tools ourselves so every call can be shown in the UI.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    thinking_config=types.ThinkingConfig(thinking_level=EFFORT.upper()) if EFFORT else None,
    max_output_tokens=16000,
)


def _context_block(question, user_id):
    """Per-turn context: today's date, relevant memories (memory RAG), indexed reports."""
    memories = rag.recall(question, user_id)
    memory_lines = "\n".join(f"- [{m['id']}] {m['fact']}" for m in memories) or "(nothing yet)"
    reports = ", ".join(f"{c} {y}" for c, y in rag.list_reports()) or "(none)"
    return (
        "<context>\n"
        f"Today: {date.today().isoformat()}\n"
        f"What you remember about this user:\n{memory_lines}\n"
        f"Annual reports indexed for search_annual_report: {reports}\n"
        "</context>"
    )


def run(client, messages, question, ctx):
    """Answer one user question.

    `messages` is the full API conversation (short-term memory); it is appended to in place.
    Yields events for the UI:
      {"type": "tool_call", "name", "input"}
      {"type": "tool_result", "name", "output", "is_error"}
      {"type": "answer", "text"}
    """
    messages.append(types.Content(role="user", parts=[
        types.Part(text=_context_block(question, ctx.user_id)),
        types.Part(text=question),
    ]))

    for _ in range(MAX_STEPS):
        response = client.models.generate_content(model=MODEL, contents=messages, config=CONFIG)
        candidate = response.candidates[0] if response.candidates else None
        parts = (candidate.content.parts if candidate and candidate.content else None) or []
        if not parts:
            # Blocked or empty reply. Still record a model turn so the conversation stays well-formed.
            reason = candidate.finish_reason.name if candidate and candidate.finish_reason else "no candidates"
            text = f"The model returned no answer ({reason}). Try rephrasing the question."
            messages.append(types.Content(role="model", parts=[types.Part(text=text)]))
            yield {"type": "answer", "text": text}
            return

        # Keep the whole content (thought signatures + function calls), not just the text.
        messages.append(candidate.content)
        text = "\n\n".join(p.text for p in parts if p.text and not p.thought).strip()
        calls = [p.function_call for p in parts if p.function_call]

        if calls:
            results = []
            for call in calls:
                args = dict(call.args or {})
                yield {"type": "tool_call", "name": call.name, "input": args}
                output, is_error = run_tool(call.name, args, ctx)
                yield {"type": "tool_result", "name": call.name, "output": output, "is_error": is_error}
                results.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response={"error": output} if is_error else {"output": output},
                )))
            # All results go back in ONE user message.
            messages.append(types.Content(role="user", parts=results))
            continue

        if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
            text += "\n\n_(The answer was cut off at the output limit.)_"
        yield {"type": "answer", "text": text or "The model returned an empty answer. Try rephrasing the question."}
        return

    yield {"type": "answer", "text": f"I stopped after {MAX_STEPS} tool steps without finishing. Try a narrower question."}


def make_client(api_key=None):
    # With no argument the SDK reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
    return genai.Client(api_key=api_key) if api_key else genai.Client()

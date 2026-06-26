"""
app/services/agent.py  — Cognitive LangGraph Agent

PHILOSOPHY:
  Previous versions used the LLM as a form parser: extract structured fields,
  check thresholds, route to search. That's not cognitive — it's bureaucratic.

  This version gives Claude the full menu and lets it reason directly, the way
  a knowledgeable waiter does. The LLM:
    1. Understands the conversation naturally
    2. Reasons about the menu from real knowledge
    3. Identifies which specific dishes match
    4. Retrieves those exact dishes via RAG for structured card data
    5. Writes a direct, organized response

  No follow-up gatekeeping. No form fields. Just: understand → match → respond.

GRAPH TOPOLOGY:
  START
    |
  [understand]  - Claude reads conversation + full menu, reasons about intent,
    |             identifies matching dish names, classifies response type
    |
  [route] ──────────────────────────────────────────► [direct_answer] ── END
    |   (specific dish / confirmation / no match)
    |
  [fetch_dishes]  - RAG fetches structured data for identified dish names
    |
  [compose_reply] - Claude writes organized response grounded in fetched data
    |
  END
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

import anthropic
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.core.config import get_settings
from app.services.rag import get_vector_store

log      = logging.getLogger(__name__)
MENU_FILE = Path(__file__).parent.parent.parent / "data" / "menu.json"


def _load_menu_context() -> str:
    """
    Build a compact, LLM-readable representation of the full menu.
    This is what the LLM reasons from — not embeddings, actual menu knowledge.
    """
    menu = json.loads(MENU_FILE.read_text())
    lines = ["THE CHEESECAKE FACTORY MENU\n"]
    by_cat: dict[str, list] = {}
    for item in menu:
        by_cat.setdefault(item["category"], []).append(item)

    for cat, items in sorted(by_cat.items()):
        lines.append(f"== {cat} ==")
        for d in items:
            dietary = ", ".join(d.get("dietary", [])) or "none"
            taste   = ", ".join(d.get("taste_profile", []))
            lines.append(
                f"  {d['name']} | ${d['price']} | {d['calories']} cal | "
                f"{d['protein_g']}g protein | taste: {taste} | dietary: {dietary}"
            )
            lines.append(f"    {d.get('description', '')}")
        lines.append("")

    return "\n".join(lines)


# Loaded once at import time — menu doesn't change during a session
MENU_CONTEXT = _load_menu_context()


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:         list[dict]   # full conversation history
    user_message:     str          # current user input
    response_type:    str          # "direct" | "menu_search"
    matched_dishes:   list[str]    # dish names identified by Claude
    retrieved_dishes: list[dict]   # structured data from RAG
    response:         str          # final text to send to client


# ── Prompts ────────────────────────────────────────────────────────────────────

UNDERSTAND_SYSTEM = f"""\
You are the AI brain of The Cheesecake Factory menu assistant.
You have complete knowledge of the menu below and respond like a knowledgeable,
experienced server — not a form or a chatbot.

{MENU_CONTEXT}

Your job when a guest sends a message:

1. UNDERSTAND what they actually want — read between the lines.
   "italian" means pasta-style dishes.
   "something spicy" means dishes with chile, hot sauce, spicy flavors.
   "light" means SkinnyLicious or lower-calorie options.
   "show me everything" means list all items in the relevant category.
   "what do you have" means browse all options.
   A specific dish name means they want details about that exact dish.

2. IDENTIFY the best matching dishes from the menu above.
   Be specific — return exact dish names from the menu.
   If nothing matches, say so honestly.

3. CLASSIFY your response type:
   - "direct": guest asked about a specific dish, said yes/confirmed,
     or asked a question not requiring menu search (e.g. "what are your hours")
   - "menu_search": guest wants to find or browse dishes

Respond ONLY with valid JSON — no markdown, no code blocks, no text before or after.
Start your response with {{ and end with }}:
{{
  "response_type": "direct" or "menu_search",
  "matched_dishes": ["exact dish name from menu", ...],
  "direct_response": "if response_type is direct: a clear, direct answer (2-3 sentences). else empty string.",
  "reasoning": "brief internal note on why you matched these dishes"
}}

Rules:
- matched_dishes must be EXACT names from the menu above. No invented names.
- For menu_search, always return at least 1 matched dish if anything remotely fits.
- For "show me italian" or "show me all desserts" — return ALL matching items.
- Never ask follow-up questions. Make your best judgment and return results.
- If truly nothing matches, return empty matched_dishes and explain in direct_response.
- Never return more than 5 matched_dishes."""


COMPOSE_SYSTEM = """\
You are a server at The Cheesecake Factory writing a clean menu recommendation.

Format each dish exactly like this, one block per dish, blank line between each:

Dish Name
[one sentence: the key fact most relevant to what they asked — calories, protein, price, taste]
[one sentence: why this dish fits what they asked for, specifically]

Rules:
- No markdown. No bold, no dashes, no bullets, no emoji.
- Dish name on its own line, then exactly two sentences.
- Blank line between dishes.
- One short closing sentence after all dishes.
- Under 160 words total.
- Sound like a knowledgeable person, not a brochure."""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def understand(state: AgentState) -> dict:
    """
    Node 1: The cognitive core.
    Claude reads the conversation + full menu and reasons directly about
    what the guest wants and which dishes match. No form parsing, no
    threshold checks — just genuine reasoning from menu knowledge.
    """
    settings = get_settings()
    client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    response = client.messages.create(
        model=settings.model_name,
        max_tokens=600,
        system=UNDERSTAND_SYSTEM,
        messages=[
            *state["messages"][:-1],
            {
                "role":    "user",
                "content": state["user_message"],
            },
        ],
    )

    text = response.content[0].text.strip()

    # Strip markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find JSON object even if Claude added text before/after it
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("understand() JSON parse failed: %s", text[:200])
        parsed = {
            "response_type":   "direct",
            "matched_dishes":  [],
            "direct_response": "I had trouble understanding that. Could you rephrase what you're looking for?",
            "reasoning":       "parse error",
        }

    log.info(
        "understand() -> type=%s dishes=%s reasoning=%s",
        parsed.get("response_type"),
        parsed.get("matched_dishes"),
        parsed.get("reasoning", "")[:100],
    )

    return {
        "response_type":  parsed.get("response_type", "direct"),
        "matched_dishes": parsed.get("matched_dishes", []),
        "response":       parsed.get("direct_response", ""),
    }


def route_after_understand(state: AgentState) -> Literal["direct_answer", "fetch_dishes"]:
    if state["response_type"] == "direct" or not state["matched_dishes"]:
        return "direct_answer"
    return "fetch_dishes"


def direct_answer(state: AgentState) -> dict:
    """
    Terminal node for direct responses — specific dish questions,
    confirmations, no-match explanations.
    Response was already generated in understand().
    """
    return {
        "response": state.get("response") or "I'm not sure about that — could you tell me more about what you're looking for?",
    }


async def fetch_dishes(state: AgentState) -> dict:
    """
    Node: Fetch structured dish data from ChromaDB for the specific
    dish names Claude identified. This is targeted retrieval — not
    broad semantic search, but fetching exact named items for card data.
    """
    store         = get_vector_store()
    matched_names = state["matched_dishes"]

    if not matched_names:
        return {"retrieved_dishes": []}

    # Use search_within to get structured data for exact dish names
    # Query is the user message — similarity within the matched set
    try:
        dishes = await store.search_within(
            query=state["user_message"],
            candidate_names=matched_names,
            n=min(5, len(matched_names)),
        )
    except Exception as e:
        log.warning("fetch_dishes search_within failed: %s", e)
        dishes = []

    # If RAG didn't return all matched dishes (e.g. not in index),
    # load missing ones directly from menu.json as fallback
    returned_names = {d.name for d in dishes}
    missing = [n for n in matched_names if n not in returned_names]

    if missing:
        menu = json.loads(MENU_FILE.read_text())
        for item in menu:
            if item["name"] in missing:
                from app.services.rag import RetrievedDish
                dishes.append(RetrievedDish(
                    name=item["name"],
                    category=item["category"],
                    calories=item["calories"],
                    protein_g=item["protein_g"],
                    carbs_g=item.get("carbs_g", 0),
                    fat_g=item["fat_g"],
                    price=item["price"],
                    dietary=item.get("dietary", []),
                    taste=item.get("taste_profile", []),
                    occasions=item.get("occasion", []),
                    description=item.get("description", ""),
                    similarity_score=1.0,
                ))

    return {
        "retrieved_dishes": [
            {
                "name":             d.name,
                "category":         d.category,
                "calories":         d.calories,
                "protein_g":        d.protein_g,
                "price":            d.price,
                "dietary":          d.dietary,
                "description":      d.description,
                "similarity_score": d.similarity_score,
            }
            for d in dishes
        ],
    }


def compose_reply(state: AgentState) -> dict:
    """
    Node: Write the final organized response grounded in fetched dish data.
    """
    settings = get_settings()
    client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    dish_context = "\n".join(
        f"- {d['name']} ({d['calories']} cal, {d['protein_g']}g protein, ${d['price']}): {d['description']}"
        for d in state["retrieved_dishes"]
    )

    response = client.messages.create(
        model=settings.model_name,
        max_tokens=400,
        system=COMPOSE_SYSTEM,
        messages=[
            *state["messages"][:-1],
            {
                "role":    "user",
                "content": (
                    f"Guest asked: {state['user_message']}\n\n"
                    f"Dishes to present:\n{dish_context}\n\n"
                    "Write the organized recommendation."
                ),
            },
        ],
    )

    return {"response": response.content[0].text.strip()}


async def compose_reply_streaming(state: AgentState) -> AsyncGenerator[str, None]:
    """Streaming version for WebSocket token delivery."""
    settings = get_settings()
    client   = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    dish_context = "\n".join(
        f"- {d['name']} ({d['calories']} cal, {d['protein_g']}g protein, ${d['price']}): {d['description']}"
        for d in state["retrieved_dishes"]
    )

    with client.messages.stream(
        model=settings.model_name,
        max_tokens=400,
        system=COMPOSE_SYSTEM,
        messages=[
            *state["messages"][:-1],
            {
                "role":    "user",
                "content": (
                    f"Guest asked: {state['user_message']}\n\n"
                    f"Dishes to present:\n{dish_context}\n\n"
                    "Write the organized recommendation."
                ),
            },
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── Graph compilation ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("understand",     understand)
    graph.add_node("direct_answer",  direct_answer)
    graph.add_node("fetch_dishes",   fetch_dishes)
    graph.add_node("compose_reply",  compose_reply)

    graph.set_entry_point("understand")

    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "direct_answer": "direct_answer",
            "fetch_dishes":  "fetch_dishes",
        },
    )

    graph.add_edge("fetch_dishes",  "compose_reply")
    graph.add_edge("direct_answer", END)
    graph.add_edge("compose_reply", END)

    return graph.compile()


GRAPH = build_graph()


# Keep for backward compat with chat.py
generate_reply_streaming = compose_reply_streaming
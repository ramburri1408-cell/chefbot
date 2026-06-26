"""
app/api/chat.py

WebSocket endpoint for the conversational chat interface.
Session state uses Redis when available, falls back to in-memory
dict when Redis is not running (local development).

Protocol:
  Client -> Server: { "message": "I want something light" }
  Server -> Client: { "type": "token", "content": "Here" }
                    { "type": "dishes", "content": [{name, calories, ...}] }
                    { "type": "done" }
  On error:         { "type": "error", "content": "message" }
"""

import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.db.redis import get_redis
from app.services.agent import GRAPH, AgentState, generate_reply_streaming

router = APIRouter()
log    = logging.getLogger(__name__)

# In-memory fallback when Redis is unavailable (local dev)
_sessions: dict = {}


async def load_session(session_id: str) -> dict:
    redis = await get_redis()
    if redis:
        try:
            raw = await redis.get(f"session:{session_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _sessions.get(session_id, {
        "messages": [], "retrieved_dishes": []
    })


async def save_session(session_id: str, state: dict) -> None:
    data = {
        "messages":         state.get("messages", []),
        "retrieved_dishes": state.get("retrieved_dishes", []),
    }
    redis = await get_redis()
    if redis:
        try:
            await redis.setex(f"session:{session_id}", 3600, json.dumps(data))
            return
        except Exception:
            pass
    _sessions[session_id] = data


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())

    log.info("New session: %s", session_id)

    await websocket.send_json({"type": "session", "session_id": session_id})
    await websocket.send_json({
        "type":    "token",
        "content": "Welcome to The Cheesecake Factory! What are you in the mood for today?",
    })
    await websocket.send_json({"type": "done"})

    try:
        while True:
            raw          = await websocket.receive_json()
            user_message = raw.get("message", "").strip()

            if not user_message:
                continue

            session = await load_session(session_id)
            session["messages"].append({"role": "user", "content": user_message})

            # Build agent state — matches new cognitive AgentState TypedDict
            state: AgentState = {
                "messages":         session["messages"],
                "user_message":     user_message,
                "response_type":    "",
                "matched_dishes":   [],
                "retrieved_dishes": [],
                "response":         "",
            }

            try:
                result = await GRAPH.ainvoke(state, config={"recursion_limit": 5})

                response_type = result.get("response_type", "menu_search")

                if response_type == "direct" or not result.get("retrieved_dishes"):
                    # Direct answer — specific dish question, confirmation, no match
                    response_text = result.get("response", "I'm not sure about that — could you rephrase?")
                    await websocket.send_json({"type": "token", "content": response_text})
                    await websocket.send_json({"type": "done"})
                    session["messages"].append({"role": "assistant", "content": response_text})

                else:
                    # Menu search — send dish cards then stream text
                    dishes = result["retrieved_dishes"]
                    await websocket.send_json({"type": "dishes", "content": dishes[:4]})

                    full_text = ""
                    async for token in generate_reply_streaming(result):
                        await websocket.send_json({"type": "token", "content": token})
                        full_text += token

                    await websocket.send_json({"type": "done"})
                    session["messages"].append({"role": "assistant", "content": full_text})
                    session["retrieved_dishes"] = dishes

                await save_session(session_id, session)

            except Exception as exc:
                log.exception("Agent error for session %s: %s", session_id, exc)
                await websocket.send_json({
                    "type":    "error",
                    "content": "Something went wrong — please try again.",
                })

    except WebSocketDisconnect:
        log.info("Session disconnected: %s", session_id)

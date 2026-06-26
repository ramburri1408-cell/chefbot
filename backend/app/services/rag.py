"""
app/services/rag.py

Two-stage menu retrieval pipeline.

STAGE 1 - Structured filter (free, deterministic):
  Detects category/course intent from the user message and filters the full
  menu to only matching items BEFORE any embedding work happens.
  e.g. "I want a dessert" -> only the 3 dessert items are candidates.
  This is a simple lookup, not a semantic problem. Using embeddings for this
  was the root cause of the wrong-category results bug.

STAGE 2 - Semantic ranking (embedding-based):
  Within the filtered candidates, rank by cosine similarity to the user's
  preference description. This is where embeddings genuinely help:
  "something light and fresh for date night" should rank SkinnyLicious Salmon
  above Edamame even though neither matches on exact keywords.

Why this is better:
  The original single-stage semantic search was trying to do two fundamentally
  different jobs with one embedding query:
    1. Category matching (structured lookup problem)
    2. Preference matching (similarity problem)
  These need different tools. Structured filters first, semantic ranking second.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings

log = logging.getLogger(__name__)

MENU_FILE = Path(__file__).parent.parent.parent / "data" / "menu.json"

# ── Category intent mapping ───────────────────────────────────────────────────
# Maps user vocabulary to exact category names in the menu.
# This is the structured lookup layer — deterministic, no embeddings needed.

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Desserts":       ["dessert", "desert", "cake", "cheesecake", "sweet treat",
                       "oreo", "chocolate dessert", "ice cream", "pudding", "mousse",
                       "red velvet", "tiramisu", "something sweet after", "end with"],
    "Appetizers":     ["appetizer", "starter", "starters", "to start", "begin with",
                       "snack", "share", "shareable", "small plate"],
    "SkinnyLicious":  ["skinny", "light", "healthy", "low calorie", "diet",
                       "watching calories", "weight", "salad", "skinnylicious"],
    "Burgers":        ["burger", "burgers", "cheeseburger", "hamburger"],
    "Pasta":          ["pasta", "noodle", "spaghetti", "linguine", "penne"],
    "Fish & Seafood": ["fish", "seafood", "salmon", "shrimp", "tuna", "lobster", "crab"],
    "Steakhouse":     ["steak", "beef", "filet", "sirloin"],
    "Entrees":        ["entree", "main course", "main dish", "chicken", "dinner"],
    "Brunch":         ["brunch", "breakfast", "morning", "avocado toast"],
    "Sandwiches":     ["sandwich", "wrap", "club"],
    "Specialties":    ["waffle", "waffles", "specialty"],
}


def detect_category(user_message: str, conversation_history: list[dict]) -> Optional[str]:
    """
    Detect explicit category/course intent from the current message and
    recent conversation history.

    Returns the exact category string to hard-filter on, or None if the
    user is expressing a preference (light, rich, spicy) rather than
    requesting a specific course or food type.

    Checks current message first (highest signal), then recent history
    (in case they said "I only want desserts" two turns ago).
    """
    # Check current message first — highest signal
    msg_lower = user_message.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            log.info("Category detected from current message: %s", category)
            return category

    # Check recent conversation history (last 4 turns)
    for turn in reversed(conversation_history[-4:]):
        content = (turn.get("content") or "").lower()
        if not content:
            continue
        for category, keywords in CATEGORY_KEYWORDS.items():
            # Stronger signal required from history to avoid false positives
            # e.g. user said "I want chicken" earlier but now wants dessert
            strong_keywords = [kw for kw in keywords if len(kw) > 5]
            if any(kw in content for kw in strong_keywords):
                log.info("Category detected from conversation history: %s", category)
                return category

    return None


def load_full_menu() -> list[dict]:
    """Load the complete menu from JSON — used for category pre-filtering."""
    if MENU_FILE.exists():
        return json.loads(MENU_FILE.read_text())
    log.warning("menu.json not found at %s", MENU_FILE)
    return []


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedDish:
    name:             str
    category:         str
    calories:         int
    protein_g:        float
    carbs_g:          float
    fat_g:            float
    price:            float
    dietary:          list[str]
    taste:            list[str]
    occasions:        list[str]
    description:      str
    similarity_score: float


# ── Vector store abstraction ──────────────────────────────────────────────────

class VectorStore(ABC):
    @abstractmethod
    async def search(self, query: str, n: int = 6) -> list[RetrievedDish]:
        ...

    @abstractmethod
    async def search_within(
        self, query: str, candidate_names: list[str], n: int = 6
    ) -> list[RetrievedDish]:
        """Semantic search restricted to a specific set of item names."""
        ...

    @abstractmethod
    async def upsert(self, dishes: list[dict]) -> int:
        ...


class ChromaStore(VectorStore):
    """
    Local ChromaDB — zero infra, great for development.
    search_within() restricts the embedding search to a pre-filtered
    candidate set, implementing Stage 2 of the two-stage retrieval.
    """

    def __init__(self) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._col = self._client.get_or_create_collection(
            name="menu",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _row_to_dish(self, meta: dict, dist: float) -> RetrievedDish:
        return RetrievedDish(
            name=meta["name"],
            category=meta["category"],
            calories=int(meta["calories"]),
            protein_g=float(meta["protein_g"]),
            carbs_g=float(meta.get("carbs_g", 0)),
            fat_g=float(meta["fat_g"]),
            price=float(meta["price"]),
            dietary=meta.get("dietary", "").split(",") if meta.get("dietary") else [],
            taste=meta.get("taste", "").split(",") if meta.get("taste") else [],
            occasions=meta.get("occasions", "").split(",") if meta.get("occasions") else [],
            description=meta.get("description", ""),
            similarity_score=round(1.0 - float(dist), 4),
        )

    async def search(self, query: str, n: int = 6) -> list[RetrievedDish]:
        count = self._col.count()
        if count == 0:
            return []
        results = self._col.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        return [
            self._row_to_dish(meta, dist)
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]

    async def search_within(
        self, query: str, candidate_names: list[str], n: int = 6
    ) -> list[RetrievedDish]:
        """
        Stage 2: semantic ranking restricted to candidate_names only.
        Uses ChromaDB's where filter to constrain the embedding search
        to only the pre-filtered items from Stage 1.
        """
        if not candidate_names:
            return []

        count = self._col.count()
        if count == 0:
            return []

        # ChromaDB where filter — only search within these item names
        where = {"name": {"$in": candidate_names}}

        try:
            results = self._col.query(
                query_texts=[query],
                n_results=min(n, len(candidate_names)),
                where=where,
            )
            return [
                self._row_to_dish(meta, dist)
                for meta, dist in zip(results["metadatas"][0], results["distances"][0])
            ]
        except Exception as e:
            log.warning("search_within failed (%s), falling back to full search", e)
            return await self.search(query, n=n)

    async def upsert(self, dishes: list[dict]) -> int:
        ids       = [f"dish_{i}" for i in range(len(dishes))]
        documents = [_dish_to_text(d) for d in dishes]
        metadatas = [_dish_to_metadata(d) for d in dishes]
        self._col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(dishes)


class PineconeStore(VectorStore):
    """Production vector store — wire in voyage-3 embeddings for prod."""

    async def search(self, query: str, n: int = 6) -> list[RetrievedDish]:
        raise NotImplementedError("Wire in voyage-3 + Pinecone query here")

    async def search_within(
        self, query: str, candidate_names: list[str], n: int = 6
    ) -> list[RetrievedDish]:
        raise NotImplementedError("Wire in voyage-3 + Pinecone filtered query here")

    async def upsert(self, dishes: list[dict]) -> int:
        raise NotImplementedError("Wire in voyage-3 + Pinecone upsert here")


# ── Two-stage retrieval ───────────────────────────────────────────────────────

async def retrieve(
    query: str,
    user_message: str,
    conversation_history: list[dict],
    max_calories: Optional[int] = None,
    min_protein_g: Optional[float] = None,
    dietary: Optional[list[str]] = None,
    n: int = 4,
) -> list[RetrievedDish]:
    """
    Main retrieval entry point — two-stage pipeline.

    Stage 1: Structured filter
      - Detect category intent (dessert/appetizer/burger/etc)
      - Load matching items directly from menu JSON
      - Apply numeric hard filters (calories, protein)
      - Apply dietary filters
      This stage is free and deterministic — no API calls, no embeddings.

    Stage 2: Semantic ranking
      - Rank the Stage 1 candidates by embedding similarity to the query
      - Returns top N most relevant dishes
      This stage only runs on the already-filtered candidates, so it
      never returns a salmon when you asked for dessert.
    """
    store = get_vector_store()
    full_menu = load_full_menu()

    # ── Stage 1: Structured filter ────────────────────────────────────────────
    category = detect_category(user_message, conversation_history)

    if category:
        candidates = [d for d in full_menu if d["category"] == category]
        log.info("Stage 1: %d candidates after category filter (%s)", len(candidates), category)
    else:
        candidates = list(full_menu)
        log.info("Stage 1: no category detected, using all %d items", len(candidates))

    # Apply numeric hard filters on structured data
    if max_calories is not None:
        filtered = [d for d in candidates if d["calories"] <= max_calories]
        if filtered:
            candidates = filtered
            log.info("Stage 1: %d candidates after calorie filter (<= %d)", len(candidates), max_calories)

    if min_protein_g is not None:
        filtered = [d for d in candidates if d["protein_g"] >= min_protein_g]
        if filtered:
            candidates = filtered
            log.info("Stage 1: %d candidates after protein filter (>= %dg)", len(candidates), min_protein_g)

    if dietary:
        for tag in dietary:
            filtered = [
                d for d in candidates
                if tag.lower() in [x.lower().strip() for x in d.get("dietary", [])]
            ]
            if filtered:
                candidates = filtered

    if not candidates:
        log.warning("Stage 1: all filters eliminated candidates, falling back to full menu")
        candidates = list(full_menu)

    # ── Stage 2: Semantic ranking within candidates ───────────────────────────
    candidate_names = [d["name"] for d in candidates]
    log.info("Stage 2: semantic ranking %d candidates for query: %s", len(candidate_names), query)

    if len(candidate_names) <= n:
        # Fewer candidates than requested — skip embedding, return all in menu order
        # (no point running embeddings if there are only 3 desserts and we want 4)
        log.info("Stage 2: skipping embedding (candidates <= n), returning all")
        return [
            RetrievedDish(
                name=d["name"],
                category=d["category"],
                calories=d["calories"],
                protein_g=d["protein_g"],
                carbs_g=d.get("carbs_g", 0),
                fat_g=d["fat_g"],
                price=d["price"],
                dietary=d.get("dietary", []),
                taste=d.get("taste_profile", []),
                occasions=d.get("occasion", []),
                description=d.get("description", ""),
                similarity_score=1.0,
            )
            for d in candidates
        ]

    return await store.search_within(query, candidate_names, n=n)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dish_to_text(dish: dict) -> str:
    dietary  = ", ".join(dish.get("dietary", [])) or "no specific dietary label"
    return (
        f"{dish['name']} is a {dish['category']} dish. "
        f"It tastes {', '.join(dish.get('taste_profile', []))}. "
        f"{dish.get('description', '')}. "
        f"It has {dish['calories']} calories, {dish['protein_g']}g protein, "
        f"{dish.get('carbs_g', 0)}g carbs, {dish['fat_g']}g fat. "
        f"Costs ${dish['price']}. "
        f"Dietary: {dietary}. "
        f"Good for: {', '.join(dish.get('occasion', []))}."
    )


def _dish_to_metadata(dish: dict) -> dict[str, Any]:
    return {
        "name":        dish["name"],
        "category":    dish["category"],
        "calories":    dish["calories"],
        "protein_g":   dish["protein_g"],
        "carbs_g":     dish.get("carbs_g", 0),
        "fat_g":       dish["fat_g"],
        "price":       dish["price"],
        "dietary":     ", ".join(dish.get("dietary", [])),
        "taste":       ", ".join(dish.get("taste_profile", [])),
        "occasions":   ", ".join(dish.get("occasion", [])),
        "description": dish.get("description", ""),
    }


def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store == "pinecone":
        return PineconeStore()
    return ChromaStore()


# Keep apply_hard_filters for backward compatibility
def apply_hard_filters(
    dishes: list[RetrievedDish],
    max_calories: Optional[int] = None,
    min_protein_g: Optional[float] = None,
    dietary: Optional[list[str]] = None,
    category: Optional[str] = None,
) -> list[RetrievedDish]:
    result = list(dishes)
    if category:
        filtered = [d for d in result if category.lower() in d.category.lower()]
        if filtered:
            result = filtered
    if max_calories is not None:
        filtered = [d for d in result if d.calories <= max_calories]
        if filtered:
            result = filtered
    if min_protein_g is not None:
        filtered = [d for d in result if d.protein_g >= min_protein_g]
        if filtered:
            result = filtered
    if dietary:
        for tag in dietary:
            filtered = [d for d in result if tag.lower() in [x.lower().strip() for x in d.dietary]]
            if filtered:
                result = filtered
    return result

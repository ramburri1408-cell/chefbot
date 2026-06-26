"""
app/api/admin.py

Vector DB admin viewer — inspect what's actually indexed in ChromaDB.
Available at:
  GET  /admin/menu          - list all indexed items
  GET  /admin/menu/search   - semantic search with query param
  GET  /admin/menu/{name}   - get a specific item by name
  POST /admin/menu/reindex  - re-run ingestion (re-embed all items)
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.rag import get_vector_store

router    = APIRouter(prefix="/admin", tags=["admin"])
MENU_FILE = Path(__file__).parent.parent.parent / "data" / "menu.json"


class MenuItemView(BaseModel):
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
    similarity_score: float | None = None


class VectorDBStats(BaseModel):
    total_items:  int
    categories:   dict[str, int]
    price_range:  dict[str, float]
    calorie_range: dict[str, int]


@router.get("/menu", summary="List all items in the vector store")
async def list_menu():
    """
    Returns every item currently indexed in ChromaDB with full metadata.
    Use this to verify what's actually in the vector store vs what's in menu.json.
    """
    store   = get_vector_store()
    # Search with a broad query to get all items
    results = await store.search("food restaurant menu", n=50)

    menu_json = json.loads(MENU_FILE.read_text())
    indexed_names = {d.name for d in results}
    all_names     = {d["name"] for d in menu_json}
    not_indexed   = all_names - indexed_names

    by_category: dict[str, list] = {}
    for d in results:
        by_category.setdefault(d.category, []).append({
            "name":        d.name,
            "calories":    d.calories,
            "protein_g":   d.protein_g,
            "price":       d.price,
            "dietary":     d.dietary,
            "similarity":  d.similarity_score,
        })

    return {
        "total_indexed":   len(results),
        "total_in_menu":   len(menu_json),
        "not_indexed":     list(not_indexed),
        "by_category":     by_category,
    }


@router.get("/menu/search", summary="Semantic search over the vector store")
async def search_menu(
    q: str = Query(..., description="Search query"),
    n: int = Query(default=5, ge=1, le=25, description="Number of results"),
):
    """
    Run a semantic search against ChromaDB and see exactly what it returns.
    Useful for debugging why certain queries return unexpected results.
    """
    store   = get_vector_store()
    results = await store.search(q, n=n)

    return {
        "query":   q,
        "results": [
            {
                "rank":        i + 1,
                "name":        d.name,
                "category":    d.category,
                "calories":    d.calories,
                "protein_g":   d.protein_g,
                "price":       d.price,
                "similarity":  round(d.similarity_score, 4),
                "description": d.description[:100] + "..." if len(d.description) > 100 else d.description,
            }
            for i, d in enumerate(results)
        ],
    }


@router.get("/menu/stats", summary="Vector store statistics")
async def menu_stats():
    """Summary statistics about what's in the vector store."""
    store   = get_vector_store()
    results = await store.search("food", n=50)

    if not results:
        return {"total": 0}

    by_cat = {}
    for d in results:
        by_cat[d.category] = by_cat.get(d.category, 0) + 1

    return VectorDBStats(
        total_items=len(results),
        categories=by_cat,
        price_range={
            "min": round(min(d.price for d in results), 2),
            "max": round(max(d.price for d in results), 2),
            "avg": round(sum(d.price for d in results) / len(results), 2),
        },
        calorie_range={
            "min": min(d.calories for d in results),
            "max": max(d.calories for d in results),
            "avg": int(sum(d.calories for d in results) / len(results)),
        },
    )


@router.get("/menu/{item_name}", summary="Get a specific menu item by name")
async def get_menu_item(item_name: str):
    """
    Fetch a specific item from the vector store by name.
    Also shows the raw menu.json entry for comparison.
    """
    store   = get_vector_store()
    results = await store.search_within(item_name, [item_name], n=1)

    menu_json = json.loads(MENU_FILE.read_text())
    raw_item  = next((d for d in menu_json if d["name"].lower() == item_name.lower()), None)

    if not results and not raw_item:
        raise HTTPException(status_code=404, detail=f"Item '{item_name}' not found")

    return {
        "vector_store": {
            "found":       len(results) > 0,
            "data":        {
                "name":        results[0].name,
                "category":    results[0].category,
                "calories":    results[0].calories,
                "protein_g":   results[0].protein_g,
                "similarity":  results[0].similarity_score,
            } if results else None,
        },
        "menu_json": raw_item,
    }


@router.post("/menu/reindex", summary="Re-index all menu items")
async def reindex_menu():
    """
    Re-runs ingestion — re-embeds all items from menu.json into ChromaDB.
    Use this after updating menu.json with new items or prices.
    """
    from app.services.ingestion import ingest
    try:
        await ingest(MENU_FILE)
        return {"status": "ok", "message": "Menu re-indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
app/models/menu.py

SQLAlchemy ORM models for the menu data layer.

Design decisions:
- Async SQLAlchemy throughout — no blocking DB calls in a FastAPI async context
- UUID primary keys — globally unique, no sequential scan on distributed writes
- updated_at trigger handled at DB level via server_default + onupdate
- Soft-delete on MenuItems (is_active flag) so removed items don't break
  existing conversation history that references them by name
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    items: Mapped[list["MenuItem"]] = relationship("MenuItem", back_populates="category")


class MenuItem(Base):
    __tablename__ = "menu_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    category_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("categories.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    # Nutrition
    calories: Mapped[int] = mapped_column(Integer, nullable=False)
    protein_g: Mapped[float] = mapped_column(Float, nullable=False)
    carbs_g: Mapped[float] = mapped_column(Float, nullable=False)
    fat_g: Mapped[float] = mapped_column(Float, nullable=False)
    fiber_g: Mapped[Optional[float]] = mapped_column(Float)
    sodium_mg: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata stored as Postgres arrays — avoids a separate junction table
    # for a read-heavy, rarely-updated reference like dietary tags
    dietary_tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    taste_profile: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    occasions: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    # Vector store reference — the embedding ID in Chroma/Pinecone
    # Kept here so we can invalidate/re-embed when price or description changes
    embedding_id: Mapped[Optional[str]] = mapped_column(String(64))
    embedding_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    category: Mapped["Category"] = relationship("Category", back_populates="items")


class Conversation(Base):
    """
    Lightweight audit log — stores summarized conversation outcomes
    for analytics (which dishes get recommended, which get ordered, etc).
    Full conversation history lives in Redis for the session duration,
    then gets archived here on session end.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    recommended_items: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

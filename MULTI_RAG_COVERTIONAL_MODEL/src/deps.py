"""Shared FastAPI dependencies."""

from collections.abc import AsyncGenerator

from db.session import get_session
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session

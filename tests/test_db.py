import pytest
import aiosqlite
from app.models.database import (
    init_db, create_image, get_image, list_images,
    Base, engine
)
from sqlalchemy import text

@pytest.fixture(autouse=True)
async def setup_db():
    # Use in-memory SQLite for testing to avoid disk locks, by mocking the engine URL 
    # Or just rely on a temp db file! We will use the default for simplicity but wrapped in a transaction or drop tables.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.mark.asyncio
async def test_create_and_get_image():
    # Insert
    created = await create_image("test.jpg", "/tmp/test.jpg", 800, 600)
    assert created is not None
    assert created["filename"] == "test.jpg"
    assert "id" in created
    
    # Retrieve
    retrieved = await get_image(created["id"])
    assert retrieved is not None
    assert retrieved["id"] == created["id"]
    assert retrieved["width"] == 800

@pytest.mark.asyncio
async def test_list_images():
    await create_image("a.jpg", "/tmp/a.jpg", 100, 100)
    await create_image("b.jpg", "/tmp/b.jpg", 200, 200)
    
    images = await list_images()
    assert len(images) == 2
    # Ensure annotation count is 0
    assert images[0]["annotation_count"] == 0

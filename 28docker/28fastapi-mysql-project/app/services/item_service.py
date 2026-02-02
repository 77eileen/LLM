# sqlalchemy를 이용한 CRUD 로직을 캡슐화
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.item import Item
from app.schemas.item import ItemCreate, ItemUpdate

class ItemService:
    @staticmethod
    async def create_item(db: AsyncSession, item_in: ItemCreate):
        db_item = Item(**item_in.model_dump())
        db.add(db_item)
        await db.commit()
        await db.refresh(db_item)
        return db_item

    @staticmethod
    async def get_items(db: AsyncSession, skip: int = 0, limit: int = 100):
        result = await db.execute(select(Item).offset(skip).limit(limit))
        return result.scalars().all()

    @staticmethod
    async def get_item(db: AsyncSession, item_id: int):
        result = await db.execute(select(Item).filter(Item.id == item_id))
        return result.scalars().first()

    @staticmethod
    async def delete_item(db: AsyncSession, item_id: int):
        db_item = await ItemService.get_item(db, item_id)
        if db_item:
            await db.delete(db_item)
            await db.commit()
        return db_item
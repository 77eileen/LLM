from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.db.session import get_db
from app.services.item_service import ItemService
from app.schemas.item import ItemCreate, ItemResponse

router = APIRouter()

@router.post("/", response_model=ItemResponse)
async def create_item(item_in: ItemCreate, db: AsyncSession = Depends(get_db)):
    return await ItemService.create_item(db, item_in)

@router.get("/", response_model=List[ItemResponse])
async def read_items(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await ItemService.get_items(db, skip=skip, limit=limit)

@router.get("/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await ItemService.get_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.delete("/{item_id}")
async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
    item = await ItemService.delete_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Successfully deleted"}
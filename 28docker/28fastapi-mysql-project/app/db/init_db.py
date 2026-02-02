from app.db.session import engine
from app.models.base import Base
from app.models.item import Item
from app.models.file_model import UploadedFile

async def init_table():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('database tables created successfully')
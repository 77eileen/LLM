from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    id = Column (Integer, primary_key=True, index =True)
    name = Column(String(200), nullable=False, index =True)
    description = Column(Text, nullable=True)
    price = Column (Float, nullable=False)
    stock = Column (Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
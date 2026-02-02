from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import DeclarativeBase

# 모든 테이블의 공통 속성을 정의
class Base(DeclarativeBase):
    pass

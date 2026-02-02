from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

# 1. 비동기 엔진 생성
# echo=True: 실행되는 SQL 쿼리를 로그에 출력 (디버깅용)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True
)

# 2. 비동기 세션 팩토리 생성
# DB 요청이 들어올 때마다 이 팩토리를 통해 세션(Session)을 찍어냅니다.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# 3. Dependency Injection (의존성 주입) 용 함수
# API 요청마다 DB 세션을 열고, 작업이 끝나면 닫아주는 역할을 합니다.
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
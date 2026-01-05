# DB 연결 정보 정의
# SQLAIchemy Engine 생성
# 세션 생성 안전한 종료 관리
# 세선 : 사용자 관련 정보가 저장 (일종의 캐시 성격을 가짐/ 쿠키.. PC에 저장되는 정보/ 서버가 알수없는 로컬만의 정보를 저장/ 민감한 정보 저장하지말것)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager   # with문으로 DB세션을 쓰기위해.

# 데이터베이스 url 설정
SQLALCHEMY_DATABASE_URL = 'sqlite:///./products.db'


# SQLite 는 기본적으로 단일 스레드 제한
# SQLite + FastAPI 조합시 다중 스레드 문제 발생
# 이를 해결하기 위한 옵션추가
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args = {"check_same_thread": False} #SQLite 특정 옵션
)



# 트랜잭션 제어
# 예외 발생시 롤백관리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# yield dbf가 endpoint 함수에 전달 --> endpoint 함수 종료시 finally 블록 실행
# 실제 호출방법
# @app.get():
# def test(db: Session = Depends(get_db)):
#     products = db.query(models.Product).all()
#     return products
def get_db():  # API처리요청
    db = SessionLocal()
    try:
        yield db   # yield: 빌려주고 회수의 개념
    finally:
        db.close()


# 파이썬이 관리하는 방식, 데이터를 스크립트로 초기화 하거나 기타 테스트코드 적용시 사용 
@contextmanager  # 라우터방식이 아닌, 일반 파이썬 방식 호출??????, 내부에서 with문 사용가능??
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
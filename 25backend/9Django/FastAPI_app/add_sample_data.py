
"""
FastAPI 데이터베이스에 샘플 데이터를 추가하는 스크립트
"""
from database import SessionLocal, engine
import models

# 테이블 생성
models.Base.metadata.create_all(bind=engine)

# 샘플 데이터
sample_products = [
    {
        "name": "삼성 갤럭시 노트북",
        "description": "고성능 업무용 노트북, 16GB RAM, 512GB SSD",
        "price": 1499000,
        "stock": 15
    },
    {
        "name": "LG 그램",
        "description": "초경량 노트북, 14인치, 배터리 수명 20시간",
        "price": 1799000,
        "stock": 8
    },
    {
        "name": "애플 맥북 프로",
        "description": "M2 칩, 13인치 레티나 디스플레이",
        "price": 2499000,
        "stock": 5
    },
    {
        "name": "로지텍 MX Master 3",
        "description": "무선 마우스, 인체공학적 디자인",
        "price": 129000,
        "stock": 50
    },
    {
        "name": "기계식 키보드",
        "description": "RGB 백라이트, 청축",
        "price": 89000,
        "stock": 30
    },
    {
        "name": "USB-C 허브",
        "description": "7in1 멀티포트 어댑터",
        "price": 45000,
        "stock": 100
    },
    {
        "name": "27인치 모니터",
        "description": "4K UHD, IPS 패널, 60Hz",
        "price": 399000,
        "stock": 12
    },
    {
        "name": "무선 이어폰",
        "description": "노이즈 캔슬링, 블루투스 5.0",
        "price": 199000,
        "stock": 25
    },
    {
        "name": "웹캠",
        "description": "1080p Full HD, 마이크 내장",
        "price": 79000,
        "stock": 40
    },
    {
        "name": "노트북 거치대",
        "description": "알루미늄 재질, 각도 조절 가능",
        "price": 35000,
        "stock": 60
    }
]

def add_sample_data():
    db = SessionLocal()
    
    try:
        # 기존 데이터 확인
        existing = db.query(models.Product).count()
        
        if existing > 0:
            print(f"이미 {existing}개의 제품이 있습니다.")
            response = input("기존 데이터를 삭제하고 새로 추가하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                db.query(models.Product).delete()
                db.commit()
                print("기존 데이터를 삭제했습니다.")
            else:
                print("작업을 취소했습니다.")
                return
        
        # 샘플 데이터 추가
        for product_data in sample_products:
            product = models.Product(**product_data)
            db.add(product)
        
        db.commit()
        print(f"\n {len(sample_products)}개의 샘플 제품을 추가했습니다!")
        
        # 추가된 데이터 확인
        print("\n 추가된 제품 목록:")
        products = db.query(models.Product).all()
        for p in products:
            print(f"  - {p.id}. {p.name} (₩{p.price:,}) - 재고: {p.stock}개")
        
    except Exception as e:
        print(f" 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 50)
    print("FastAPI 샘플 데이터 추가 스크립트")
    print("=" * 50)
    print()
    add_sample_data()
    print()
    print("완료! FastAPI 서버를 시작하고 Django에서 확인하세요.")
    print("  - FastAPI: http://localhost:8001/docs")
    print("  - Django: http://localhost:8000")

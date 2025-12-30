
"""
FastAPI 데이터베이스에 샘플 사용자 및 제품 데이터를 추가하는 스크립트
"""
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from auth import get_password_hash

# 데이터베이스 테이블 생성
models.Base.metadata.create_all(bind=engine)


def add_sample_users(db: Session):
    """샘플 사용자 추가"""
    users_data = [
        {
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "관리자",
            "role": "admin",
            "password": "admin123"
        },
        {
            "username": "manager",
            "email": "manager@example.com",
            "full_name": "매니저",
            "role": "manager",
            "password": "manager123"
        },
        {
            "username": "user1",
            "email": "user1@example.com",
            "full_name": "사용자1",
            "role": "user",
            "password": "user123"
        },
        {
            "username": "user2",
            "email": "user2@example.com",
            "full_name": "사용자2",
            "role": "user",
            "password": "user123"
        }
    ]
    
    created_users = []
    for user_data in users_data:
        # 이미 존재하는지 확인
        existing_user = db.query(models.User).filter(
            models.User.username == user_data["username"]
        ).first()
        
        if not existing_user:
            password = user_data.pop("password")
            hashed_password = get_password_hash(password)
            
            user = models.User(
                **user_data,
                hashed_password=hashed_password,
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            created_users.append(user)
            print(f" 사용자 생성: {user.username} (role: {user.role})")
        else:
            created_users.append(existing_user)
            print(f"  이미 존재하는 사용자: {existing_user.username}")
    
    return created_users


def add_sample_products(db: Session, users: list):
    """샘플 제품 추가"""
    products_data = [
        {
            "name": "노트북",
            "description": "고성능 업무용 노트북",
            "price": 1500000,
            "stock": 10,
            "owner_id": users[0].id  # admin
        },
        {
            "name": "무선 마우스",
            "description": "인체공학적 디자인 무선 마우스",
            "price": 35000,
            "stock": 50,
            "owner_id": users[1].id  # manager
        },
        {
            "name": "키보드",
            "description": "기계식 키보드 - 청축",
            "price": 120000,
            "stock": 25,
            "owner_id": users[2].id  # user1
        },
        {
            "name": "모니터",
            "description": "27인치 4K 모니터",
            "price": 450000,
            "stock": 15,
            "owner_id": users[2].id  # user1
        },
        {
            "name": "웹캠",
            "description": "HD 화상회의용 웹캠",
            "price": 85000,
            "stock": 30,
            "owner_id": users[3].id  # user2
        },
        {
            "name": "헤드셋",
            "description": "노이즈 캔슬링 헤드셋",
            "price": 150000,
            "stock": 20,
            "owner_id": users[3].id  # user2
        }
    ]
    
    for product_data in products_data:
        # 이름으로 중복 체크
        existing_product = db.query(models.Product).filter(
            models.Product.name == product_data["name"]
        ).first()
        
        if not existing_product:
            product = models.Product(**product_data)
            db.add(product)
            db.commit()
            db.refresh(product)
            owner = db.query(models.User).filter(models.User.id == product.owner_id).first()
            print(f" 제품 생성: {product.name} (owner: {owner.username})")
        else:
            print(f"  이미 존재하는 제품: {existing_product.name}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("샘플 데이터 추가 시작")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # 사용자 추가
        print("\n[사용자 데이터 추가]")
        users = add_sample_users(db)
        
        # 제품 추가
        print("\n[제품 데이터 추가]")
        add_sample_products(db, users)
        
        print("\n" + "=" * 60)
        print("샘플 데이터 추가 완료!")
        print("=" * 60)
        print("\n 생성된 계정 정보:")
        print("-" * 60)
        print("관리자 계정 - ID: admin, PW: admin123")
        print("매니저 계정 - ID: manager, PW: manager123")
        print("사용자 계정1 - ID: user1, PW: user123")
        print("사용자 계정2 - ID: user2, PW: user123")
        print("-" * 60)
        
    except Exception as e:
        print(f"\n 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()


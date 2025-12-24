# pydantic으로 데이터 검증하기
# 복잡한 데이터를 안전하게 받고 검증
from fastapi import FastAPI
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# 검증규칙이란 Pydantic BaseModel + Field에 정의한 “타입 + 필수 여부 + 범위 조건”의 집합
# BaseModel pydantic의 기본 클래스
# Field 검증 규칙 추가
# name:str = Field(...min_length=1, max_length=100)
# price:int = Field(...,gt=0)
# ... 필수항목

# gt(greater than) ~ 보다 큼(초과)
# ge(greater equal) ~ 이상
# lt(less than) ~ 보다 작다(미만)
# le(less equal) ~ 이하
# pattern : 정규표현식 패턴
# Optinal : 선택...(없어도 됨)

app = FastAPI(
    title='pydantic 데이터 검증',
    description='데이터 모델 정의 및 검증',
    version='0.0.1'
)

class Item(BaseModel):
    '''상품 정보 모델'''
    name:str
    price:int

@app.post('/item/simple')
def create_item_simple(item: Item):
    '''
    상품추가 api

    ex):
    {
        'name' : '노트북',
        'price' : 1000000
    }
    '''
    return {
        'mesaage':f"{item.name}이(가) 추가되었습니다.",
        'item':item
    }  

# 검증 규칙이 있는 모델
class Product(BaseModel):
    '''검증 규칙이 포함된 상품 모델'''
    name:str = Field(...,min_length=1, max_length=100,description='상품명') # ... 필수표시
    price:int = Field(...,gt=0,description='가격(0보다 커야함)')
    description: Optional[str] = Field(None, max_length=500,description='상품설명')  # Optional 옵션
    is_stock:bool=Field(default=True,description='재고여부')
    class Config:
        json_schema_extra={
            'example':{
                'name':'노트북',
                'price':1000000,
                'description':'인체공학적 디자인',
                'is_stock':True
            }        
        }
@app.post('/products')
def create_product(product:Product):
    '''검증이 포함된 상품 추가 api
    검증규칙:
    -name : 
    '''
    return {
        'message':f"{product.name}이(가) 추가되었습니다.",
        'product':product
    }

# GET은 “조회”, POST는 “생성/전송”이다.
# GET은 상태를 바꾸지 말아야 하고, POST는 상태를 바꾼다.
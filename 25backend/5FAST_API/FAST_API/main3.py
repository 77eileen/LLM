from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


app = FastAPI(
    title = 'todo crud basic',
    description = '',
    version = '1.0.0'
)

# 데이터 모델 정의
# 클라이언트가 POST 요청 시 보내는 데이터 검증
# Field로 최소/최대 길이, 설명 추가
# ... 은 필수
class TodoCreate(BaseModel):
    '''기본모델 정보'''
    title:str = Field(...,min_length=1, max_length=100, description='할일 제목')
    class Config:
        json_schema_extra={
            'example' :{
                'title': '할일 제목 넣기'
        }}

# 서버가 응답할 데이터 형식 정의
# response_model로 지정하면 자동 검증 + 문서화
class TodoResponse(BaseModel):
    '''서버가 반환하는 정보'''
    id:int
    title:str
    completed:bool
    created_at:str
    class Config:
        json_schema_extra={
            'example' :{
                'id' : 1,
                'title': '제목',
                'completed': False,
                'created_at': '2025-12-22 14:25-25'
        }}

# 글로벌 함수
todos = []
next_id =1

# 라우터
# 메인페이지
@app.get('/')
def index():
    '''메인페이지'''
    return{
        'message':'메인페이지'
    }

# 추가 - todo 생성
@app.post('/todos',response_model=TodoResponse,status_code=status.HTTP_201_CREATED) # status_code 성공하면 201번이 뜸
def create_todo(todo:TodoCreate):
    '''새로운 todo 추가
    :Args:
        todo : todo 제목
    :Returns:
        생성된 todo
    '''
    global next_id # next_id가 지역변수에서 적용되니까 글로벌로 적용되도록 
    new_todo = {
        'id':next_id,
        'title':todo.title,
        'completed':False,
        'created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    #리스트에 추가
    todos.append(new_todo)
    #id증가
    next_id += 1
    return new_todo


# 조회는 전체조회와 id 별로 조회 두가지로 가능

# 전체 조회
@app.get('/todos', response_model=list[TodoResponse])
def get_all_todos():
    '''모든 todo조회'''
    return todos

# id 별로 조회 / id값을 어떻게 전달하는지에 따라 경로vs 쿼리로 나눠진다
@app.get('/todos/{id}', response_model=TodoResponse)
# @app.get('/find_todos', response_model=TodoResponse)  # http://127.0.0.1:8080/todos?id=2
def get_todo(id:int):
    '''특정 id로 찾기'''
    for todo in todos:
        if todo['id'] == id:
            return todo
    raise HTTPException(    # 에러났을 때 알려주는 것
        status_code=status.HTTP_404_NOT_FOUND,
        detail = f'id {id} 해당하는 데이터가 없습니다.'
    )
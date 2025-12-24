# CORS cross-origin resource sharing
# 다른 도메인(프론트엔드)에서 api 호출

# 만들 aip
# POST   /todos                할 일 추가
# GET    /todos                전체 조회
# GET    /todos/{id}           단일 조회
# PUT    /todos/{id}           수정
# DELETE /todos/{id}           삭제
# GET /todos/filter/completed 완료목록
# GET /todos/filter/pending 미완료목록
# DELETE /todos/clear/all      전체 삭제
# GET    /health               서버 상태

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # CORS -> 다른 도메인에서 이 API를 호출할 수 있게 허용하는 규칙
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


app=FastAPI(
    title='todo crud',
    description="추가 등록 수정 삭제",
    version='2.0.0'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],    # 모든 도메인 허용
    allow_credentials=True, # 쿠키/인증정보 포함 요청
    allow_methods=['*'],    # 모든 http method  (get post delete put ...)
    allow_headers=['*']     # 모든 헤더
)

# 데이터 모델(입력과 출력의 형태 제한)
class TodoCreate(BaseModel):
    '''todo생성'''
    # 실제DB와 연결시.. DB에 설정된 정보가 있으면 같이 맞춰주면 좋음
    title:str = Field(...,min_length=1,max_length=100,description='할일 제목')      # title 필수
    description:Optional[str] = Field(None,max_length=500,description='할일 설명')  # description 선택

class TodoResponse(BaseModel):
    '''Todo 응답'''
    id:int
    title:str
    description:Optional[str]
    completed:bool
    created_at:str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    modified_at:Optional[str] = None

class TodoUpdate(BaseModel):
    '''todo 수정 - 모든 필드는 선택적'''    # Field(...) 쓰면 필수처럼 동작/ 옵션이면 Field(None, ...) 써야 한다
    title:str = Field(...,min_length=1, max_length=100)
    description:Optional[str] = Field(None,max_length=500)
    completed:Optional[bool] = None
    modified_at:str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 저장소
todos_db = []
next_id = 0

# api 앤드 포인트 ->> api가 바라보는 마지막 방향
@app.get('/')
def index():
    return '메인 페이지'

# 추가
@app.post('/todos',response_model=TodoResponse,status_code=status.HTTP_201_CREATED)
def create_todo(todo:TodoCreate):
    '''todo 생성
    
    request body:
    {
        'title': '할일 제목',
        'description' : '할일 설명 -선택' 
    }'''

    global next_id
    next_id += 1
    new_data = TodoResponse(
        id=next_id,
        title=todo.title,
        description=todo.description,
        completed=False,
    )
    todos_db.append(new_data)
    return new_data

# 전체 조회
@app.get('/todos',response_model=list[TodoResponse])
def get_all_todos():
    '''모든 데이터 조회'''
    return todos_db

# 아이디별 데이터 조회
@app.get('/todos/{id}',response_model=TodoResponse)
def get_todo_by_id(id:int):
    '''아이디로 조회'''
    for todo in todos_db:
        if todo.id == id:
            return todo
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f'{id}를 찾을수 없습니다.'
    )
# 수정
@app.put('/todos/{id}',response_model=TodoResponse)
def update_todo(id:int, update_data:TodoUpdate):
    '''수정'''
    # 목록에서 id에 해당하는 요소를 찾아서 값을 변경
    get_todo_by_id(id)    
    for todo in todos_db:
        if todo.id == id:
            print(f'수정 {update_data}')            
            todo.modified_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if update_data.title is not None:
                todo.title = update_data.title
            if update_data.description is not None:
                todo.description = update_data.description
            if update_data.completed is not None:
                todo.completed = update_data.completed                
            return todo
    
# 삭제
@app.delete('/todos/{id}',status_code=status.HTTP_204_NO_CONTENT)
def delete_todo(id:int):
    '''삭제'''
    get_todo_by_id(id)

    for index, todo in enumerate(todos_db):
        if todo.id== id:
            todos_db.pop(index)
            return

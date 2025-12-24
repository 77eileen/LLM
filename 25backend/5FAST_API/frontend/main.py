# Step 5: 프론트엔드 연동
# 목표: HTML + jQuery와 FastAPI 백엔드 연결하기

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from pathlib import Path

# 현재 파일의 위치를 기준으로 절대 경로 설정
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# 1. FastAPI 앱 생성 및 설정
app = FastAPI(
    title="Step 5: TODO with Frontend",
    description="HTML + jQuery 프론트엔드와 연동",
    version="5.0.0"
)

# 2. CORS 설정 -> 브라우저가 “다른 출처(origin)”로의 요청을 허용할지 말지를 결정하는 규칙
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 어떤 주소에서 오는 요청이든 허용하겠다는 의미
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 정적 파일 경로 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# http://localhost:8000/static/app.js 같은 주소로 요청이 오면, 실제 컴퓨터의 static 폴더 안에 있는 app.js 파일을 클라이언트에게 보내주라는 의미
# ============================================
# 4. 데이터 모델 정의 (pydantic)
# ============================================
class TodoCreate(BaseModel):    # 클라이언트가 '할일 생성'을 요청할 때 title과 description을 보내야 한다는 규칙
    title: str = Field(..., min_length=1, max_length=200) # ... 은 필수라는 뜻
    description: Optional[str] = Field(None, max_length=1000)

class TodoUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    completed: Optional[bool] = None

class TodoResponse(BaseModel):  # 서버가 클라이언트에게 응답을 보낼 때, 아래와 같은 정보를 포함해야 한다는 규칙
    id: int
    title: str
    description: Optional[str]
    completed: bool
    created_at: str

# ============================================
# 5. 데이터 저장소 (인메모리 DB)
# ============================================
todos_db = []   # 서버를 끄면 모든 데이터가 사라지는 임시 저장소
next_id = 1

# ============================================
# HTML 페이지 제공
# ============================================
@app.get("/")
def read_index():
    """메인 HTML 페이지 반환"""
    return FileResponse(str(STATIC_DIR / "index.html"))

# ============================================
# API 엔드포인트
# ============================================
# app.js가 새로운 할일 데이터를 보내오면(POST방식), 이를 받아 todos_db에 추가하고 결과를 반환
@app.post("/api/todos", response_model=TodoResponse, status_code=status.HTTP_201_CREATED)
def create_todo(todo: TodoCreate):
    """TODO 추가"""
    global next_id
    
    new_todo = {
        "id": next_id,
        "title": todo.title,
        "description": todo.description,
        "completed": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    todos_db.append(new_todo)
    next_id += 1
    
    print(f"[CREATE] TODO 추가: {new_todo['title']}")
    return new_todo

# 모든 할일 목록(todos_db)을 조회하여 반환
@app.get("/api/todos", response_model=list[TodoResponse])
def get_all_todos():
    """TODO 목록 조회"""
    print(f"[READ] TODO 조회: {len(todos_db)}개")
    return todos_db


@app.get("/api/todos/{todo_id}", response_model=TodoResponse)
def get_todo(todo_id: int):
    """특정 TODO 조회"""
    for todo in todos_db:
        if todo["id"] == todo_id:
            return todo
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"ID {todo_id}인 TODO를 찾을 수 없습니다"
    )


@app.put("/api/todos/{todo_id}", response_model=TodoResponse)
def update_todo(todo_id: int, todo_update: TodoUpdate):
    """TODO 수정"""
    for todo in todos_db:
        if todo["id"] == todo_id:
            if todo_update.title is not None:
                todo["title"] = todo_update.title
            if todo_update.description is not None:
                todo["description"] = todo_update.description
            if todo_update.completed is not None:
                todo["completed"] = todo_update.completed
            
            print(f"[UPDATE] TODO 수정: ID={todo_id}")
            return todo
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"ID {todo_id}인 TODO를 찾을 수 없습니다"
    )


@app.delete("/api/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_todo(todo_id: int):
    """TODO 삭제"""
    for index, todo in enumerate(todos_db):
        if todo["id"] == todo_id:
            todos_db.pop(index)
            print(f"[DELETE] TODO 삭제: ID={todo_id}")
            return
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"ID {todo_id}인 TODO를 찾을 수 없습니다"
    )


@app.get("/api/health")
def health_check():
    """서버 상태 및 통계"""
    total = len(todos_db)
    completed = len([t for t in todos_db if t["completed"]])
    pending = total - completed
    
    return {
        "status": "healthy",
        "total": total,
        "completed": completed,
        "pending": pending
    }



# 서버 시작 시 디렉토리 확인
@app.on_event("startup")
def startup_event():
    print(f"📁 BASE_DIR: {BASE_DIR}")
    print(f"📁 STATIC_DIR: {STATIC_DIR}")
    print(f"✅ static 폴더 존재: {STATIC_DIR.exists()}")
    if STATIC_DIR.exists():
        print(f"📄 static 폴더 내용: {list(STATIC_DIR.iterdir())}")


# ============================================
# 실행 방법:
# uvicorn main:app --reload
# 
# 브라우저에서:
# http://localhost:8000  ← HTML UI
# http://localhost:8000/docs  ← API 문서
# ============================================
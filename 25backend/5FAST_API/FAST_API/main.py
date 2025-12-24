# uvicorn main:app --reload --> 파일 실행 명령어

# 라우터: 경로를 지정해주는 것. 데코레이터 문법으로 라우터 기능을 구현
# 데코레이터 @app.get('/hello') --- http://localhost:8000/hello -> 원래 로컬 경로  -> 서버내의 해당 함수를 실행


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# fast API 앱 키   --> 이 서버는 어떤 API인지 설명하는 메타데이터
app = FastAPI(
    title = 'FAST API',
    description = '기본기능 확인',
    version = '0.0.1'
)
# /static으로 시작하는 모든 요청을 static/ 폴더의 파일을 찾아서 응답하는 전용 앱에 넘긴다
# app.mount("/static", StaticFiles(directory="static"), name="static") 

# @app.get('/hello')
# def say_hello():
#     return {'message' : 'Hello world'}

@app.get('/')       # '/' 는 기본 주소. 쉽게 말하면 메인페이지
def index():
    '''브라우저에서 http://localhost:8000 접속 시 실행'''
    return {'message' : '첫번째 화면'}                  # 자동으로 json형태로 변환

@app.get('/hello')      # 함수 매개변수 = 요청 데이터
def say_hello(name:str, lang:str):
    '''두개의 파라메터를 쿼리스트링으로 전달  http://localhost:8000/hello?name=홍길동&lang=ko'''
    if lang=='ko':
        return {'message': f'안녕하세요 {name}'}  
    elif lang=='en':
        return {'message': f'Hello {name}'}
    else:
        return "lang 정보를 입력하세요  ?name=이름&lang=언어"


# 경로 파라미터 vs 쿼리 파라미터

# 경로 파라메터
# http://localhost:8000/hello/홍길동
@app.get('/hello/{name}')
def say_hello(name:str):
    return f"안녕하세요 {name}"       # 자동으로 json 형태로 변환
# => Hello 홍길동    이 출력

# 쿼리 파라미터
# http://localhost:8000/greet?name=홍길동&age=35  / 물음표 앞까지는 경로
@app.get('/greet')  
def greet(name:str, age:int):
    return f'반갑습니다, {name}님 당신의 나이는 {age}입니다'


# /multiply 엔드포인트를 만들어서 두 숫자를 입력 받아서 곱해서 출력하기
@app.get('/multiply')  
def multiply(num1:int, num2:int):
    return num1*num2
    
# http://127.0.0.1:8000/docs
#-> 전체 내가 만든 함수?기능들 다 볼 수 있음


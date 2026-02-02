from fastapi import FastAPI

# FastAPI 앱
app = FastAPI(
    title='FAST API',
    description="기본기능 확인",
    version='0.0.1'
)

@app.get('/')
def index():
    '''브라우져에서 http://localhost:8080 접속시 실행'''
    return {'message': '첫번째 화면'}

# 데코레이터 @app.get('/hello')  --- 라우터   http://localhost:8000/hello  --> 서버내의 해당 함수를 실행
# @app.get('/hello')
@app.get('/hello')
def say_hello(name:str, lang:str):
    '''두개의 파라메터를 쿼리스트링으로 전달  http://localhost:8000/hello?name=홍길동&lang=ko'''
    if lang=='ko':
        return {'message': f'안녕하세요 {name}'}  # 자동으로 json형태로 변환
    elif lang=='en':
        return {'message': f'Hello {name}'}
    else:
        return "lang 정보를 입력하세요  ?name=이름&lang=언어"

# 경로 파라메터 vs 쿼리 파라메터
#  http://localhost:8000/hello/홍길동   경로파라메터
@app.get('/hello/{name}')    
def say_hello(name: str):
    return f"안녕하세요 {name}"
    

@app.get('/greet')    # http://localhost:8000/greet?name=홍길동&age=35
def greet(name: str, age: int):
    return f"반갑습니다. {name}님 당신의 나이는 {age}입니다."


@app.get('/multiply')
def multiply(num1: int, num2: int):
    return num1 * num2


# /multiply 엔드포인트를 만들어서 두 수자를 입력받아서 곱해서 출력하기
# /hello/{name} 에 쿼리 파라메터로 lang 을 추가해서
    # lang=ko -> 안녕하세요
    # lang=en -> hello




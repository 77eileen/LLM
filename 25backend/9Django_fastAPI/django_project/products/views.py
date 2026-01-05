from django.shortcuts import render, redirect
from django.conf import settings
import httpx
from .forms import *
from django.contrib import messages
from asgiref.sync import sync_to_async, async_to_sync  # 👈 추가



# Create your views here.
# 비동기(즉시 주지않음. await로 기다려야함)로 FastAPI와 연동해야함


FASTAPI_URL = settings.FASTAPI_BASE_DIR

# 전체 제품 조회
async def get_products():
    async with httpx.AsyncClient() as client:  # 비동기 http 커넥션 (django 3.1 이상에서 가능)
        try :
            response = await client.get(f"{FASTAPI_URL}/api/products")
            response.raise_for_status() # 오류 발생시 예외 발생
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error fetching products: {e}")
            return []
    

# ID에 대한 제품 조회 (파이썬은 오버로딩 안되므로 위에 전체 제품 조회 함수와 함수명이 달라야함)
async def get_products_detail(product_id):
    async with httpx.AsyncClient() as client:  # 비동기 http 커넥션 (django 3.1 이상에서 가능)
        try :
            response = await client.get(f"{FASTAPI_URL}/api/products/{product_id}")
            response.raise_for_status() # 오류 발생시 예외 발생
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error fetching products: {e}")
            return None


# 제품 등록
async def create_product(data):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{FASTAPI_URL}/api/products", json=data)
            response.raise_for_status()  
            return True
        except httpx.HTTPError as e:
            print(f"Error creating product: {e}")
            return False
        

# 제품 업데이트
async def update_product(product_id, data):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(f"{FASTAPI_URL}/api/products/{product_id}", json=data)
            response.raise_for_status()  
            return True
        except httpx.HTTPError as e:
            print(f"Error creating product: {e}")
            return False
        

# 제품 삭제
async def delete_product(product_id):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(f"{FASTAPI_URL}/api/products/{product_id}")
            response.raise_for_status()  
            return True
        except httpx.HTTPError as e:
            print(f"Error creating product: {e}")
            return False


        
# FastAPI 연결...?
#####################################################################################################
# Django 자체를 이용...?

async def product_list(request):
    products = await get_products()
    return render(request, 'products/product_list.html', {'products': products})


# product_create 경로에, 2가지 방식 
    # 1단계: 사용자가 "상품 등록" 버튼 클릭
    # → GET 요청: /products/create/
    # → 빈 폼을 보여줌 (입력 준비)

    # 2단계: 사용자가 폼에 데이터 입력 후 "저장" 버튼 클릭
    # → POST 요청: /products/create/
    # → 데이터를 받아서 처리 (저장)
async def product_create(request):
    if request.method =='GET':
        form = ProductForm()
    elif request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            # 폼에서 데이터 추출
            data = form.cleaned_data
            result = await create_product(data)
            if result:
                messages.success(request, '제품이 성공적으로 생성되었습니다.')
                return redirect ('products:product_list') # url 별칭
            else:
                messages.error(request, '제품 생성에 실패했습니다.')
    return render(request, 'products/product_form.html', {'form': form, 'title': '제품등록'})



# 제품 수정
async def product_edit(request, product_id):
    # ID로 제품을 조회 후 사용자가 전달한 값으로 업데이트 FastAPI 요청
    product = await get_products_detail(product_id)
    if not product:
        messages.error(request, '제품을 찾을 수 없습니다.')
        return redirect('products:product_list')
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            result = await update_product(product_id, data)
            if result:
                messages.success(request, '제품이 성공적으로 수정되었습니다.')
                return redirect('products:product_list')
            else:
                messages.error(request, '제품 수정에 실패했습니다.')
    else:
        form = ProductForm(initial=product)  # form을 호출하면서 product 값으로 초기화
    return render(request, 'products/product_form.html'
                  , {'form': form, 'title': '제품수정'})




async def product_delete(request, product_id):
    if request.method == 'POST':
        result = await delete_product(product_id)
        if result:
            messages.success(request, '제품이 성공적으로 삭제되었습니다.')
        else:
            messages.error(request, '제품 삭제에 실패했습니다.')

    return redirect('products:product_list')



# 👉 render = 화면을 “그려서 바로 응답” : 
        # 서버가 바로 화면을 그려서 내보냄 (URL 변경없음)
        # 보통 GET은 render 
# 👉 redirect = “다른 URL로 다시 요청하라고 지시” : (CRUD는 redirect가 유리?)
        # 서버가 URL말고 저기로 다시가/ 새로운 GET요청을 다시 보냄
        # 보통 POST는 redirect 



####################### 인증관련 ####################################################

######## 인증관련 fastAPI 연동? ########

async def register_user(data):
    async with httpx.AsyncClient() as client: # 비동기 http 커넥션
        try:
            response = await client.post(f"{FASTAPI_URL}/api/auth/register", json=data)
            response.raise_for_status()  # 오류발생시 예외 발생
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error register user: {e}")
            return None


async def login_user(data):
    """로그인"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{FASTAPI_URL}/api/auth/token",
                data={
                    "username": data["username"],
                    "password": data["password"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError as e:
            print(f"Error login user: {e}")
            return None


async def get_current_user(token:str):
    '''현재 로그인한 사용자 정보 조회'''
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{FASTAPI_URL}/api/auth/me", 
                                        headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()  # 오류발생시 예외 발생
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error get current user: {e}")
            return None





######## 인증관련 django ########

def register_view(request):
    """회원가입"""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # ✅ first_name + last_name을 full_name으로 합치기
            payload = {
                'username': form.cleaned_data['username'],
                'email': form.cleaned_data['email'],
                'password': form.cleaned_data['password'],
                'full_name': f"{form.cleaned_data.get('first_name', '')} {form.cleaned_data.get('last_name', '')}".strip(),  # 👈 수정
                'role': 'user'  # 기본값
            }
            
            print(f"📤 Django → FastAPI 전송 데이터: {payload}")
            
            result = async_to_sync(register_user)(payload)
            
            print(f"📥 FastAPI 응답: {result}")
            
            if result:
                messages.success(request, '회원가입이 완료되었습니다.')
                return redirect('login')
            else:
                messages.error(request, '회원가입에 실패했습니다.')
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form, 'title': '회원가입'})



async def login_view(request):
    """로그인"""
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            result = await login_user(form.cleaned_data)
            
            if result:
                # 👇 sync_to_async로 감싸서 세션 저장
                await sync_to_async(request.session.__setitem__)('access_token', result['access_token'])
                
                messages.success(request, '로그인되었습니다.')
                return redirect('products:product_list')
            else:
                messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
    else:
        form = UserLoginForm()
    
    return render(request, 'registration/login.html', {'form': form, 'title': '로그인'})



# def logout_view(request):
#     '''로그아웃'''

            

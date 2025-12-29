from django.shortcuts import render, redirect
from django.conf import settings
import httpx
from .forms import ProductForm
from django.contrib import messages




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
    # post 방식으로 fastapi 정의한것처럼 수행
    # get 방식
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
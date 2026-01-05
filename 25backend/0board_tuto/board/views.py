from django.shortcuts import render,get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.utils import timezone
from .models import *
from .forms import *


# Create your views here.

# 모든 뷰 함수는 request 파라미터를 받습니다.
# request : 사용자가 보낸 요청(request) 전체 정보가 들어있는 객체로 브라우저가 서버에게 보낸 모든 정보 묶음
# 요청(request)은 언제 생기나
    # 브라우저 주소창에 http://127.0.0.1:8000/ 입력
    # 링크 클릭
    # 폼 제출
    # 버튼 클릭 (AJAX 포함)
    # ➡️ HTTP 요청(Request) 이 서버로 날아감
    # ➡️ Django가 HttpRequest 객체 생성
    # ➡️ 그걸 request로 뷰에 전달
# request 안에 뭐가 들어 있나?
    # request.method ====> 요청방식 (GET/POST)
    # request.GET =======> URL 파라미터 (URL에서 ? 뒤에 붙는 값)
    # request.POST ======> form submit (html에서 form method ="post"로 전송된 값들이 전부 request.POST에 들어있음)
    # request.user ======> 로그인한 사용자 정보
    # request.session ===> 세션정보
    # request.META ======> 브라우저/헤더정보
# def index(request):
#     '''게시판 메인 페이지'''
#     return render(request, 'board/index.html')  # 템플릿 파일을 렌더링 하여 응답으로 변환/ Django는 자동으로 각 앱의 templates 폴더에서 템플릿을 찾음

def post_list(request):
    '''게시글 목록 페이지'''
    posts = Post.objects.all() # Post 모델의 모든 객체를 가져옴
    today = timezone.now().date() # 게시글이 오늘 작성되었는지 표시
    for post in posts:
        post.is_new = post.created_at.date() == today
    context = {'posts': posts} # 템플릿에 전달할 컨텍스트
    return render(request, 'board/post_list.html', context) # 템플릿 파일을 렌더링

def post_detail(request, pk):
    '''게시글 상세 페이지
    PK: 게시글 ID'''
    post = get_object_or_404(Post, pk=pk) # 객체를 가져오고, 없으면 404 에러 표시

    # 조회수 증가
    post.increase_views()

    context = {'post': post} # 템플릿에 전달할 컨텍스트
    return render(request, 'board/post_detail.html', context) # 템플릿 파일을 렌더링

@login_required # 로그인한 사용자만 접근가능
def post_create(request):
    """게시글 작성 페이지"""
    if request.method == 'POST':
        form = PostForm(request.POST)  # 제출된 데이터로 폼 생성
        if form.is_valid(): # 유효성 검사
            post = form.save(commit=False) # 임시저장(DB에 저장하지 않음)
            post.author = request.user #작성자 설정 (현재 로그인한 사용자)
            post.save() # DB에 저장
            messages.success(request, '게시글이 성공적으로 작성되었습니다.')
            return redirect('board:detail', pk=post.pk)
    else:
        form = PostForm()  # 빈폼 생성

    context = {
        'form': form,
    }
    return render(request, 'board/post_form.html', context)


@login_required
def post_update(request, pk):
    """게시글 수정 페이지"""
    post = get_object_or_404(Post, pk=pk)

    # 작성자만 수정 가능
    if post.author != request.user: # 작성자 확인
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('board:detail', pk=post.pk)

    if request.method == 'POST':
        form = PostForm(request.POST, instance=post) # instance=post: 기존 게시글 데이터를 폼에 채움
        if form.is_valid():
            form.save()
            messages.success(request, '게시글이 수정되었습니다.')
            return redirect('board:detail', pk=post.pk)
    else:
        form = PostForm(instance=post)

    context = {
        'form': form,
        'post': post,
    }
    return render(request, 'board/post_form.html', context)


@login_required
def post_delete(request, pk):
    """게시글 삭제"""
    post = get_object_or_404(Post, pk=pk)

    # 작성자만 삭제 가능
    if post.author != request.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('board:detail', pk=post.pk)

    if request.method == 'POST': # post요청일때만 실제 삭제
        post.delete()
        messages.success(request, '게시글이 삭제되었습니다.')
        return redirect('board:list')

    context = {
        'post': post,
    }
    return render(request, 'board/post_confirm_delete.html', context) # get 이면 삭제확인 페이지ㄴ

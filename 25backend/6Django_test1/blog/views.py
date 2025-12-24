from django.shortcuts import render, redirect
from .models import Post
# Create your views here.



# ws.post_list, name='post_list'),
    # path('<int:pk>/', views.post_detail, name='post_detail'),
    # path('create/', views.post_create, name='post_create'),
    # path('<int:pk>/update/', views.post_update, name='post_update'),
    # path('<int:pk>/delete/', views.post_delete,


def post_list(request):
    '''모든 포스트 리스트를 보여준다'''
    # 데이터베이스에서 post 테이블의 데이터를 전부 가져와서 html에 전달
    # DB를 담당하는 모델을 부른다
    posts = Post.objects.all()
    content = {
        'posts':posts,
    }
    return render(request, 'blog/post_list.html', content)

def post_detail(request):
    pass

def post_create(request):
    '''화면에서 post 방식으로 전달한 데이터를 가지고 post 테이블에 저장'''
    if request.method == 'POST':
        title=request.POST['title']  # html에서 name = title
        content=request.POST['content'] # html에서 name = content
        post = Post(title=title, content=content)
        post.save() #insert 쿼리가 실행
        return redirect('blog:post_list')  # DB가 갱신(CRUD)되면 새로운 DB에 정보를 가지고 화면을 refresh.. 그래서 redirect..? (blog의 host name을 호출..???)
    elif request.method == 'GET':
        return render(request, 'blog/post_form.html')

def post_update(request):
    pass

def post_delete(request):
    pass
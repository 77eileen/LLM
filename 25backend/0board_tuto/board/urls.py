from django.urls import path
from . import views  # from . import views: 현재 폴더(board)의 views.py를 import 합니다.


app_name = 'board' # 나중에 URL을 참조할 때 사용할 네임스페이스입니다.

urlpatterns = [
    # path('', views.index, name='index'), # 빈 경로('')로 접속하면 views.index 함수를 실행합니다.
    path('', views.post_list, name='list'), # 빈 경로('')로 접속하면 게시글 목록
    path('post/<int:pk>/', views.post_detail, name='detail'), # <int:pk> 정수형 파라미터로 게시글 ID
    path('post/create/', views.post_create, name='create'),
    path('post/<int:pk>/update/', views.post_update, name='update'),
    path('post/<int:pk>/delete/', views.post_delete, name='delete'),
]


# Django는 URL패턴을 위애서 아래로 순서대로 검사
# 첫 번째로 매칭되는 패턴을 사용해서 뷰(View)를 호출함
# 매칭되면 끝, 아래 패턴은 더 이상 검사하지 않음
# 첫 번째 패턴 'post/<int:pk>/' 검사 → <int:pk> 자리에 "create" 들어가 있음
# 그런데 <int:pk>는 정수만 매칭 가능 → "create"는 정수 아니므로 404 오류 발생 ❌
# 즉, post_create 뷰가 호출되지 않음
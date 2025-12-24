from django.urls import path, include
from . import views

app_name = 'blog'

urlpatterns = [
    # path('blog/', views.index), #http://127.0.0.1:8000/blog/blog/
    path('', views.index), #http://127.0.0.1:8000/blog/
    path("<int:question_id>/", views.detail, name='detail'),  # alies 별칭 name = 이름 상세페이지
    path('register/answer/<int:question_id>/', views.answer_create, name='answer_create'),
]
# <> 는 변하는 값(파라미터) 이 들어올 자리이라는 뜻
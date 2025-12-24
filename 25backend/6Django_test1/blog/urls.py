# blog/urls.py

from django.urls import path
from . import views

app_name = 'blog'  # 네임스페이스 : 앱을 여러개 만들었기 때문에 구분하기 위해서 만듬.

urlpatterns = [
    path('', views.post_list, name='post_list'),   # http://127.0.0.1:8000/blog/
    # path('<int:pk>/', views.post_detail, name='post_detail'),
    path('create/', views.post_create, name='post_create'),
    # path('<int:pk>/update/', views.post_update, name='post_update'),
    # path('<int:pk>/delete/', views.post_delete, name='post_delete'),
]
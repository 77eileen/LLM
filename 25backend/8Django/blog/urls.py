# 파이썬에서 라우터는?
# Router는 ViewSet의 액션들을 자동으로 URL 패턴으로 변환해주는 도구

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'blog'

# Router 생성
router = DefaultRouter()
# ViewSet 등록
router.register(r'posts', views.PostViewSet, basename='posts')  # /api/posts/my_posts
router.register(r'comments', views.CommentViewSet, basename='comment')
router.register(r'categories', views.CategoryViewSet, basename='category')
router.register(r'tags', views.TagViewSet, basename='tag')

urlpatterns = [
    path('', include(router.urls) ),
]


# 자동생성되는 url
# http method       URL               Action              Name
# GET               /posts/           list                post-list
# POST              /posts/           create              post-list
# GET               /posts/{pk}/      retrieve            post-detail
# PUT               /posts/{pk}/      update              post-detail
# PATCH             /posts/{pk}/      partial_update      post-detail
# DELETE            /posts/{pk}/      destroy             post-detail
from django.contrib import admin
from .models import Category, Tag, Post, Comment, Bookmark, Like


# Register your models here.
# 하기 작성하면 관리자 화면에서 하기 사항이 생긴것을 확인할 수 있음

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']


@admin.register(Tag)
class TagsAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']


@admin.register(Post)
class PostsAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'category', 'views', 'published', 'created_at']
    list_filter = ['published', 'category', 'created_at']
    search_fields = ['title', 'author__username', 'content']
        # author__username : FK 관계를 타고 검색 후 join 쿼리를 수행
    filter_horizontal = ['tags']  # 위젯 표현 방식 : 좌우 이동가능한
    readonly_fields = ['created_at', 'updated_at', 'views']
    
    # 레이아웃 정의
    fieldsets = (
        ('기본 정보', {
            'fields': ('title', 'author', 'category')
        }),
        ('내용', {
            'fields': ('content', 'tags')
        }),
        ('상태', {
            'fields': ('published', 'views')
        }),
        ('날짜', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['author', 'post', 'created_at']
    list_filter = ['created_at']
    search_fields = ['author__username', 'post__title', 'content']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(Like)
class LikeAdmin(admin.ModelAdmin):
    list_display = ['user', 'post', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__username', 'post__title']


@admin.register(Bookmark)
class BookmarkAdmin(admin.ModelAdmin):
    list_display = ['user', 'post', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__username', 'post__title']
from django.contrib import admin
from .models import *
# Register your models here.


@admin.register(Post) # Post 모델을 관리자 페이지에 등록
class PostAdmin(admin.ModelAdmin):
    '''게시글 관리자 페이지 설정'''
    list_display = ['title', 'author', 'created_at', 'updated_at'] # 목록에 표시될 필드
    list_filter = ['created_at', 'author'] # 필터 옵션
    search_fields = ['title', 'content'] # 검색 옵션
    readonly_fields = ['created_at', 'updated_at', 'views'] # 읽기 전용 필드
    date_hierarchy = 'created_at' # 날짜별로 필터링할수있는 네비게이션
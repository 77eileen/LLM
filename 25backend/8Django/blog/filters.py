from django_filters import rest_framework as filters
from .models import Post, Tag

class PostFilter(filters.FilterSet):
    # 제목 부분 일치 검색
    title = filters.CharFilter(lookup_expr='icontains')
    # 내용 부분 일치 검색
    content = filters.CharFilter(lookup_expr='icontains')
    # 날짜 범위 필터
    created_after = filters.DateRangeFilter(field_name='created_at', lookup_expr='gte') # gte 이상
    created_before = filters.DateRangeFilter(field_name='created_at', lookup_expr='lte')
    # 조회수 범위 필터
    min_views = filters.NumberFilter(field_name='views', lookup_expr='gte')
    max_views = filters.NumberFilter(field_name='views', lookup_expr='lte')
    # 다중 선택 필터
    filters.ModelMultipleChoiceFilter(
        field_name='tags__id',
        to_field_name = 'id',
        queryset=Tag.objects.all()
    )

    class Meta:
        model = Post
        fields = {
            'status': ['exact'],
            'category': ['exact'],
            'author':['exact'],
        }
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


# Create your models here.
# 모델을 만들고 난 뒤 실제 DB에 테이블을 생성해야하므로, 마이그레이션 진행

class Post (models.Model):
    '''
    게시글 모델
    '''
    title = models.CharField(max_length=200, verbose_name='제목') 
    content = models.TextField(verbose_name='내용') # TextField 길이 제한이 없어서 긴 글을 쓸수 있음
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='작성자') # Django가 기본으로 제공하는 사용자모델 User // 사용자가 삭제되면 그 사용자의 게시글도 모두 삭제
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='작성일시')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일시')
    views = models.PositiveIntegerField (default=0, verbose_name='조회수') # 양의 정수

    class Meta :
        ordering = ['-created_at'] # 최신 글이 위로 오도록 정렬 / -(마이너스)는 내림차순을 의미
        verbose_name = '게시글'
        verbose_name_plural = '게시글'

    def __str__(self):
        return self.title
    
    def get_absolute_url(self): # 게시글의 URL을 반환 (나중에 사용)
        '''게시글의 상세 페이지 URL을 반환'''
        return reverse("board:detail", kwargs={"pk": self.pk})
    
    def increase_views(self):
        '''조회수를 증가시키고 저장'''
        self.views += 1
        self.save(update_fields=['views'])

    def is_updated(self):
        '''수정 여부 확인 (작성 후 1분 이상 지나고 수정된 경우)'''
        return (self.updated_at - self.created_at).total_seconds() > 60

from django.db import models
from django.contrib.auth.models import User

# Create your models here.

# ORM : 쿼리대신 DB를 사용하는 방법 (장고를 테이블로 인식)
# 카테고리 모델 만들기
class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['-created_at'] # 정렬방식은 생성된 날짜 기준으로
    def __str__(self):
        return self.name
    

class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.name
    
class Post(models.Model):
    title=models.CharField(max_length=200)
    content=models.TextField()
    # FK 해주면, 다대일 관계 (Many to One : 사용자 1명 - 포스트 여러개 작성)
        #on_delete=models.CASCADE 부모키가 삭제되면 자식키도 삭제

    # related_name='posts' 역참조 할 때 사용하는 이름/ User -> Post 목록에 접근할 때 사용하는 이름
        # 역참조 : FK를 가지고 있지 않은 쪽에서 나를 참조하는 애들을 찾음
        # 설정안하면 : user.post_set.all()
        # 설정하면: user.posts.all() “이 유저가 작성한 포스트들 전부 보여줘”
    author=models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    # on_delete=models.SET_NULL, null=True ===> 부모키가 삭제되면 fk는 null로
    # category.posts.all() ==> 카테고리A에 해당하는게 누구냐..? 찾을때 사용(역참조)

    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True,
                                 blank=True, related_name='posts')
    # 다대다
    tags = models.ManyToManyField(Tag, blank=True, related_name='posts')
    created_at = models.DateTimeField(auto_now_add=True) # 생성시간 딱 한번만
    updated_at = models.DateTimeField(auto_now=True) # 수정 시간 자동갱신
    published = models.BooleanField(default=False)
    views = models.IntegerField(default=0) # 조회수
    class Meta:
        ordering = ['-created_at']
        def __str__(self):
            return self.title    


class Comment (models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    content = models.TextField()
    created_at = models.DateField(auto_now_add=True)
    updated_at = models.DateField(auto_now=True)
    class Meta:
        ordering = ['-created_at']
    def __str__(self):
        return f"{self.author.username}'s comment"
    
class Bookmark(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='bookmarks')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bookmarks')
    created_at= models.DateField(auto_now_add=True)
    class Meta:
        unique_together = ('post', 'user')  # 여러필드를 묶어서 하나의 유니크 제약조건
            # 같은 유저는 같은 포스트를 한번만 북마크 할 수 있다
    def __str__(self):
        return f"{self.user.username}'s bookmark"


class Like(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='likes')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='likes')
    created_at= models.DateField(auto_now_add=True)
    class Meta:
        unique_together = ('post', 'user')  
    def __str__(self):
        return f"{self.user.username} - {self.post.title}"
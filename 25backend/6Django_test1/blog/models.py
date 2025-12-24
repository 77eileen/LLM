from django.db import models

# Create your models here.
class Post(models.Model):
    '''블로그 포스트 모델'''
    title= models.CharField(max_length=100)
    content = models.TextField()

    def __str__(self):
        return f'[{self.pk}]{self.title}'
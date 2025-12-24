from django.db import models

# Create your models here.
# class 이름 자체가 table이 되고, model에 상속받음

class Question(models.Model):
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_at = models.DateTimeField()
    def __str__(self):
        return self.subject

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE) # on_delete=models.CASCADE 부모지워지면, 자식도 지워짐
    content = models.TextField()
    create_at = models.DateTimeField()
    
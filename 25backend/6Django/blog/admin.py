from django.contrib import admin
from .models import Question, Answer
# Register your models here.
# 하기 작성하면 관리자 화면에서 하기 사항이 생긴것을 확인할 수 있음
admin.site.register(Question)
admin.site.register(Answer)

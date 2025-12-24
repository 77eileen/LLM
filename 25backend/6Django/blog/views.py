from django.shortcuts import render
from django.http import HttpResponse
from .models import Question

# Create your views here.
# def index(request):
#     return HttpResponse('<h1> 안녕하세요 blog입니다.</h1')   # 문자열을 http로 변경해줌

def index(request):
    question_list = Question.objects.order_by('-create_at')
    context = {'question_list': question_list}
    return render(request, 'blog/question_list.html', context)


def detail(request, question_id):
    question = Question.objects.get(id=question_id)
    context = {'question': question}
    return render(request, 'blog/question_detail.html', context)

from django.utils import timezone
from django.shortcuts import redirect
def answer_create(request, question_id):  # url로 넘어온 데이터
    question = Question.objects.get(id=question_id)
    question.answer_set.create(content=request.POST.get('content'), create_at=timezone.now())
    return redirect('blog:detail', question_id=question_id)

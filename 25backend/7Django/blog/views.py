from django.shortcuts import render, get_object_or_404, redirect
from .models import *
from .forms import *
from django.views.generic import CreateView, UpdateView, DeleteView, ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy  
    # reverse_lazy??? URL을 “나중에” 해석하는 reverse

from django.db.models import Q

# Create your views here.

def post_list(request):
    posts = Post.objects.filter(published=True).select_related('author', 'category')
    return render(request, 'blog/post_list.html', {'posts':posts})
    # 상기 내용이 하기 쿼리와 동일함
    # select
    # --
    # --
    # from post
    # inner join user
    #     on user.id = post.author
    # inner join category
    #     on category.id = category
    # where post.published =1
    


def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)    
    post.views += 1
    post.save()


    # 댓글 comment
    comments = post.comments.all().select_related('author')
    context = {
        'post': post,
        'comments': comments,
        'is_liked': Like.objects
            .filter(post=post, user=request.user)
            .exists() 
                if request.user.is_authenticated else False,
                    # request.user.is_authenticated : 인증여부 판단, 즉, 인증받은 사람만 가능
        'is_bookmarked': Bookmark.objects
            .filter(post=post, user=request.user)
            .exists() 
                if request.user.is_authenticated else False,
    }
    return render(request, 'blog/post_detail.html', context)



# LoginRequiredMixin 인증되지 않은 사용자의 접근을 차단
    # 인증되지 않은 사용자가 접근하면, 로그인 페이지로 리다이렉트
    # 로그인 성공하면, 로그인 성공 후 원래 요청한 url로 다시 이동
    # LoginRequiredMixin, CreateView ==> 순서 중요. 바뀌면 안됨. 인증된 사용자가 create 할수 있으므로.
# Post Create - Class-Based View
class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    form_class = PostForm
    template_name = 'blog/post_form.html'
    success_url = reverse_lazy('blog:post_list')
    
    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


# Post Update
class PostUpdateView(LoginRequiredMixin, UpdateView):
    model = Post
    form_class = PostForm
    template_name = 'blog/post_form.html'
    success_url = reverse_lazy('blog:post_list')
    
    def get_queryset(self):
        return Post.objects.filter(author=self.request.user)


# Post Delete
class PostDeleteView(LoginRequiredMixin, DeleteView):
    model = Post
    template_name = 'blog/post_confirm_delete.html'
    success_url = reverse_lazy('blog:post_list')
    
    def get_queryset(self):
        return Post.objects.filter(author=self.request.user)


# Add Comment
@login_required
def add_comment(request, pk):
    post = get_object_or_404(Post, pk=pk)
    
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.post = post
            comment.author = request.user
            comment.save()
    
    return redirect('blog:post_detail', pk=post.pk)


# Toggle Like
@login_required
def toggle_like(request, pk):
    post = get_object_or_404(Post, pk=pk)
    like, created = Like.objects.get_or_create(post=post, user=request.user)
    
    if not created:
        like.delete()
    
    return redirect('blog:post_detail', pk=post.pk)


# Toggle Bookmark
@login_required
def toggle_bookmark(request, pk):
    post = get_object_or_404(Post, pk=pk)
    bookmark, created = Bookmark.objects.get_or_create(post=post, user=request.user)
    
    if not created:
        bookmark.delete()
    
    return redirect('blog:post_detail', pk=post.pk)


# Category Posts
def category_posts(request, pk):
    category = get_object_or_404(Category, pk=pk)
    posts = category.posts.filter(published=True)
    return render(request, 'blog/post_list.html', {'posts': posts, 'category': category})


# Search
def search(request):
    query = request.GET.get('q', '') # ?q=검색어
    posts = Post.objects.filter(published=True)
    
    if query:
        posts = posts.filter(
            Q(title__icontains=query) |
            Q(content__icontains=query) |
            Q(author__username__icontains=query)
        )
    
    return render(request, 'blog/search.html', {'posts': posts, 'query': query})


# Dashboard
@login_required
def dashboard(request):
    my_posts = Post.objects.filter(author=request.user)
    my_comments = Comment.objects.filter(author=request.user)
    my_likes = Like.objects.filter(user=request.user)
    my_bookmarks = Bookmark.objects.filter(user=request.user)
    
    context = {
        'my_posts': my_posts,
        'my_comments': my_comments,
        'my_likes': my_likes,
        'my_bookmarks': my_bookmarks,
    }
    return render(request, 'blog/dashboard.html', context)







'''
class PostCreateView(LoginRequiredMixin, CreateView): 
    pass
class PostUpdateView(LoginRequiredMixin, UpdateView):
    pass
class PostDeleteView(LoginRequiredMixin, DeleteView):
    pass

@login_required
def add_comment(request, pk):
    pass
@login_required
def toggle_like(request, pk):
    pass
@login_required
def toggle_bookmark(request, pk):
    pass

# 카테고리 테이블의 값 추출
def category_posts(request, pk):
    pass

def search(request):
    pass

@login_required
def dashboard(request):
    pass
'''
from django import forms
from .models import Post, Comment

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content', 'category', 'tags', 'published']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '포스트 제목을 입력하세요'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '포스트 내용을 입력하세요',
                'rows': 10
            }),
            'category': forms.Select(attrs={
                'class': 'form-control'
            }),
            'tags': forms.CheckboxSelectMultiple(),
            'published': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '댓글을 입력하세요',
                'rows': 3
            }),
        }

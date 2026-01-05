from django import forms
from .models import Post

# forms.ModelForm: 모델을 기반으로 폼을 자동 생성
class PostForm(forms.ModelForm):
    '''게시글 작성 폼'''
    class Meta:
        model = Post
        fields = ['title', 'content']
        # widget : form field가 HTML에서 어떻게 보여질지 결정하는 도구
        # 사용자가 입력하는 UI요소(input, textarea, select 등)를 지정
        widgets = {   
            'title': forms.TextInput(  # TextInput 한줄 입력
                attrs={   # attrs HTML 속성 설정
                    'class': 'form-control',
                    'placeholder': '제목을 입력하세요.',
                    'maxlength': 200,
                }),
            'content': forms.Textarea(   # Textare 여러 줄 입력 가능 (줄바꿈 포함)
                attrs={
                    'class': 'form-control',
                    'placeholder': '내용을 입력하세요.',
                    'rows': 10,
                    }),
        }        
                
        labels = {
            'title': '제목',
            'content': '내용',
        }   

    def clean_title(self):
        '''제목 필드의 유효성 검사'''
        title = self.cleaned_data.get('title') # self.cleaned_data → Form이 내부적으로 만든 검증 완료된 필드 값들이 들어있는 dict-like객체
        if len(title) < 2:
            raise forms.ValidationError("제목은 2글자 이상 입력해주세요 :-)")
        return title
    
    def clean_content(self):
        '''내용 필드의 유효성 검사'''
        content = self.cleaned_data.get('content')
        if len(content) < 10:
            raise forms.ValidationError("내용은 10글자 이상 입력해주세요 :-)")
        return content
                                        
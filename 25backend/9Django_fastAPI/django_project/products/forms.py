# 화면 구성을 클래스 형태로 구현해서 html에서 표시 ----.html을 대신
from django import forms
from django.contrib.auth import get_user_model


class ProductForm(forms.Form):
    name = forms.CharField(max_length=200, label='제품명'
                    , widget=forms.TextInput(attrs={'placeholder': '제품명을 입력하세요'})
                    )
    
    description = forms.CharField(
            required=False, label='제품설명'
            , widget=forms.Textarea(attrs={'rows':4, 'placeholder': '제품설명을 입력하세요'}))
    price = forms.CharField(
            required=False, label='제품가격'
            , widget=forms.NumberInput(attrs={'min':0,'step':0.01, 'placeholder': '제품가격을 입력하세요'}))
    stock = forms.IntegerField(
            initial=0, label='제품재고'
            ,widget=forms.NumberInput(attrs={'min':0, 'placeholder': '제품재고를 입력하세요'})) 




##################################### 인증 ###########################
User = get_user_model()

class UserRegistrationForm(forms.Form):  # Form 보다 발전된게 ModelForm
        '''회원가입 폼'''
        password = forms.CharField(label='비밀번호',
                        widget=forms.PasswordInput(attrs={'placeholder': '비밀번호를 입력하세요'}))
        password_confirm = forms.CharField(label='비밀번호 확인',
                        widget=forms.PasswordInput(attrs={'placeholder': '비밀번호를 입력하세요'}))
        username = forms.CharField(label='사용자명',
                                widget=forms.TextInput(attrs={'placeholder':'사용자명을 입력하세요'}))
        email = forms.CharField(label='이메일',
                                widget=forms.EmailInput(attrs={'placeholder':'이메일을 입력하세요'}))    
        first_name = forms.CharField(label='이름',
                                widget=forms.TextInput(attrs={'placeholder':'이름을 입력하세요'}))
        last_name = forms.CharField(label='성',
                                widget=forms.TextInput(attrs={'placeholder':'성을 입력하세요'}))
        

class UserLoginForm(forms.Form):
       '''로그인 폼'''
       username = forms.CharField(label='사용자명',
                        widget=forms.TextInput(attrs={'placeholder': '사용자명을 입력하세요'}))
       password = forms.CharField(label='비밀번호',
                        widget=forms.PasswordInput(attrs={'placeholder': '비밀번호를 입력하세요'}))
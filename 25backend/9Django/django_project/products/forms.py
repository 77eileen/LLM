# 화면 구성을 클래스 형태로 구현해서 html에서 표시 ----.html을 대신
from django import forms

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
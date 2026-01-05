# Part 5: 사용자 인증 기능

## 📌 이번 Part에서 배울 내용

- 회원가입 기능
- 로그인/로그아웃 기능
- 로그인 상태에 따른 화면 표시
- 사용자별 권한 설정
- 비밀번호 변경 기능
- 회원 탈퇴 기능
- Django 비밀번호 검증 규칙 이해 및 커스터마이징

## 1. accounts 앱 생성

사용자 인증 기능을 담당할 새로운 앱을 만들어봅시다.

```bash
python manage.py startapp accounts
```

### 앱 등록

**myboard_project/settings.py**에서 앱을 등록하세요:

```python
# myboard_project/settings.py

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'board',
    'accounts',  # 추가
]
```

### 로그인 관련 설정 추가

같은 파일에서 아래 내용을 추가하세요 (파일 맨 아래에 추가):

```python
# myboard_project/settings.py

# 로그인 성공 후 이동할 URL
LOGIN_REDIRECT_URL = '/'

# 로그아웃 성공 후 이동할 URL
LOGOUT_REDIRECT_URL = '/'

# 로그인이 필요한 페이지 접근 시 이동할 URL
LOGIN_URL = '/accounts/login/'
```

## 2. 회원가입 폼 만들기

**accounts/forms.py** 파일을 만들고 다음 내용을 작성하세요:

```python
# accounts/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
    """회원가입 폼"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': '이메일 주소',
        })
    )
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '사용자 이름 (ID)',
        })
    )
    password1 = forms.CharField(
        label='비밀번호',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': '비밀번호',
        })
    )
    password2 = forms.CharField(
        label='비밀번호 확인',
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': '비밀번호 확인',
        })
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def clean_email(self):
        """이메일 중복 확인"""
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('이미 사용 중인 이메일입니다.')
        return email

    def save(self, commit=True):
        """사용자 저장"""
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user
```

### 코드 설명

- `UserCreationForm`: Django가 제공하는 회원가입 폼 상속
- `password1`, `password2`: 비밀번호와 비밀번호 확인 (Django가 자동으로 일치 여부 검사)
- `clean_email()`: 이메일 중복 검사
- `save()`: 사용자 정보를 데이터베이스에 저장

### 📘 Django 비밀번호 검증 규칙

Django는 기본적으로 안전한 비밀번호를 위해 여러 검증 규칙을 적용합니다:

1. **UserAttributeSimilarityValidator**: 비밀번호가 사용자 이름, 이메일 등과 너무 유사하면 안 됨
2. **MinimumLengthValidator**: 최소 8자 이상 (기본값)
3. **CommonPasswordValidator**: 흔히 사용되는 비밀번호(예: "password", "12345678") 사용 불가
4. **NumericPasswordValidator**: 숫자로만 이루어진 비밀번호 사용 불가

이러한 검증은 `settings.py`의 `AUTH_PASSWORD_VALIDATORS` 설정에서 관리됩니다.

### 비밀번호 검증 규칙 커스터마이징 (선택사항)

개발 환경이나 특수한 경우 비밀번호 검증 규칙을 변경할 수 있습니다.

**myboard_project/settings.py**에서 다음과 같이 수정할 수 있습니다:

```python
# myboard_project/settings.py

# 기본 설정 (변경하지 않으면 이 설정이 적용됨)
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,  # 최소 길이 변경 가능 (예: 10으로 변경)
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# 개발 환경에서 간단한 비밀번호 사용하기 (운영 환경에서는 절대 사용하지 마세요!)
# AUTH_PASSWORD_VALIDATORS = []  # 모든 검증 비활성화
```

**주의**: 운영(production) 환경에서는 반드시 강력한 비밀번호 검증을 유지하세요!

## 3. 인증 뷰 만들기

**accounts/views.py**에 다음 코드를 작성하세요:

```python
# accounts/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from django.contrib import messages
from .forms import SignUpForm


def signup(request):
    """회원가입"""
    if request.user.is_authenticated:
        messages.info(request, '이미 로그인되어 있습니다.')
        return redirect('board:list')

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # 회원가입 후 자동 로그인
            messages.success(request, f'{user.username}님, 회원가입을 환영합니다!')
            return redirect('board:list')
    else:
        form = SignUpForm()

    context = {
        'form': form,
    }
    return render(request, 'accounts/signup.html', context)


def user_login(request):
    """로그인"""
    if request.user.is_authenticated:
        messages.info(request, '이미 로그인되어 있습니다.')
        return redirect('board:list')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'{user.username}님, 환영합니다!')
                # next 파라미터가 있으면 해당 페이지로, 없으면 메인으로
                next_url = request.GET.get('next', 'board:list')
                return redirect(next_url)
    else:
        form = AuthenticationForm()

    context = {
        'form': form,
    }
    return render(request, 'accounts/login.html', context)


@login_required
def user_logout(request):
    """로그아웃"""
    logout(request)
    messages.success(request, '로그아웃되었습니다.')
    return redirect('board:list')


@login_required
def profile(request):
    """프로필 페이지"""
    context = {
        'user': request.user,
    }
    return render(request, 'accounts/profile.html', context)


@login_required
def change_password(request):
    """비밀번호 변경"""
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # 세션 유지
            messages.success(request, '비밀번호가 성공적으로 변경되었습니다.')
            return redirect('accounts:profile')
    else:
        form = PasswordChangeForm(request.user)

    context = {
        'form': form,
    }
    return render(request, 'accounts/change_password.html', context)


@login_required
def delete_account(request):
    """회원 탈퇴"""
    if request.method == 'POST':
        password = request.POST.get('password')
        user = authenticate(username=request.user.username, password=password)

        if user is not None:
            # 비밀번호가 맞으면 계정 삭제
            username = user.username
            user.delete()
            messages.success(request, f'{username}님, 회원 탈퇴가 완료되었습니다.')
            return redirect('board:list')
        else:
            # 비밀번호가 틀리면 에러 메시지
            messages.error(request, '비밀번호가 올바르지 않습니다.')
            return redirect('accounts:delete_account')

    return render(request, 'accounts/delete_account.html')
```

### 코드 설명

#### signup 함수

- `if request.user.is_authenticated`: 이미 로그인된 사용자는 회원가입 페이지 접근 불가
- `login(request, user)`: 회원가입 후 자동 로그인
- `messages.success()`: 환영 메시지 표시

#### user_login 함수

- `AuthenticationForm`: Django 기본 로그인 폼
- `authenticate()`: 사용자 인증
- `login(request, user)`: 세션에 사용자 정보 저장
- `request.GET.get('next')`: 로그인 전에 접근하려던 페이지로 리다이렉트

#### change_password 함수

- `update_session_auth_hash()`: 비밀번호 변경 후에도 로그인 상태 유지

#### delete_account 함수

- `@login_required`: 로그인한 사용자만 접근 가능
- `authenticate()`: 비밀번호 확인을 위해 사용자 재인증
- `user.delete()`: 사용자 계정 삭제 (관련된 게시글도 함께 삭제됨)
- 보안을 위해 비밀번호 확인 후 탈퇴 처리

## 4. URL 패턴 설정

**accounts/urls.py** 파일을 만들고 다음 내용을 작성하세요:

```python
# accounts/urls.py

from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('password/', views.change_password, name='change_password'),
    path('delete/', views.delete_account, name='delete_account'),  # 추가
]
```

### 프로젝트 urls.py에 연결

**myboard_project/urls.py**를 수정하세요:

```python
# myboard_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('board.urls')),
    path('accounts/', include('accounts.urls')),  # 추가
]
```

## 5. 회원가입 템플릿

**accounts/templates/accounts** 폴더를 만들고 템플릿 파일들을 작성합니다.

```bash
# Windows
mkdir accounts\templates\accounts

# Mac/Linux
mkdir -p accounts/templates/accounts
```

**accounts/templates/accounts/signup.html**:

```html
<!-- accounts/templates/accounts/signup.html -->

{% extends 'board/base.html' %}

{% block title %}회원가입 - Django 게시판{% endblock %}

{% block content %}
<div style="max-width: 500px; margin: 0 auto;">
    <h2 style="text-align: center; margin-bottom: 30px;">회원가입</h2>

    <form method="post">
        {% csrf_token %}

        {% if form.non_field_errors %}
            <div style="background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 15px; margin-bottom: 20px;">
                {{ form.non_field_errors }}
            </div>
        {% endif %}

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                사용자 이름 (ID) <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.username }}
            {% if form.username.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.username.errors }}
                </div>
            {% endif %}
            <small style="color: #7f8c8d;">영문, 숫자, @/./+/-/_ 만 가능합니다.</small>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                이메일 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.email }}
            {% if form.email.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.email.errors }}
                </div>
            {% endif %}
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                비밀번호 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.password1 }}
            {% if form.password1.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.password1.errors }}
                </div>
            {% endif %}
            <small style="color: #7f8c8d;">
                • 최소 8자 이상<br>
                • 숫자로만 이루어질 수 없습니다<br>
                • 너무 흔한 비밀번호는 사용할 수 없습니다
            </small>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                비밀번호 확인 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.password2 }}
            {% if form.password2.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.password2.errors }}
                </div>
            {% endif %}
        </div>

        <button type="submit" style="width: 100%; background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 10px;">
            회원가입
        </button>
    </form>

    <div style="text-align: center; margin-top: 20px; padding-top: 20px; border-top: 1px solid #ecf0f1;">
        <p style="color: #7f8c8d;">
            이미 계정이 있으신가요?
            <a href="{% url 'accounts:login' %}" style="color: #3498db; text-decoration: none;">로그인</a>
        </p>
    </div>
</div>

<style>
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
        font-family: inherit;
    }

    .form-control:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
</style>
{% endblock %}
```

## 6. 로그인 템플릿

**accounts/templates/accounts/login.html**:

```html
<!-- accounts/templates/accounts/login.html -->

{% extends 'board/base.html' %}

{% block title %}로그인 - Django 게시판{% endblock %}

{% block content %}
<div style="max-width: 500px; margin: 0 auto;">
    <h2 style="text-align: center; margin-bottom: 30px;">로그인</h2>

    <form method="post">
        {% csrf_token %}

        {% if form.non_field_errors %}
            <div style="background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 15px; margin-bottom: 20px;">
                {{ form.non_field_errors }}
            </div>
        {% endif %}

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                사용자 이름 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.username }}
            {% if form.username.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.username.errors }}
                </div>
            {% endif %}
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                비밀번호 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.password }}
            {% if form.password.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.password.errors }}
                </div>
            {% endif %}
        </div>

        <button type="submit" style="width: 100%; background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 10px;">
            로그인
        </button>
    </form>

    <div style="text-align: center; margin-top: 20px; padding-top: 20px; border-top: 1px solid #ecf0f1;">
        <p style="color: #7f8c8d;">
            계정이 없으신가요?
            <a href="{% url 'accounts:signup' %}" style="color: #3498db; text-decoration: none;">회원가입</a>
        </p>
    </div>
</div>

<style>
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
        font-family: inherit;
    }

    .form-control:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }

    input[name="username"], input[name="password"] {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
    }

    input[name="username"]:focus, input[name="password"]:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
</style>
{% endblock %}
```

## 7. 프로필 템플릿

**accounts/templates/accounts/profile.html**:

```html
<!-- accounts/templates/accounts/profile.html -->

{% extends 'board/base.html' %}

{% block title %}내 프로필 - Django 게시판{% endblock %}

{% block content %}
<div style="max-width: 600px; margin: 0 auto;">
    <h2 style="margin-bottom: 30px;">내 프로필</h2>

    <div style="background-color: #ecf0f1; padding: 30px; border-radius: 8px;">
        <div style="margin-bottom: 20px;">
            <label style="display: block; color: #7f8c8d; margin-bottom: 5px;">사용자 이름</label>
            <p style="font-size: 1.2em; font-weight: bold;">{{ user.username }}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; color: #7f8c8d; margin-bottom: 5px;">이메일</label>
            <p style="font-size: 1.1em;">{{ user.email|default:"등록된 이메일이 없습니다." }}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; color: #7f8c8d; margin-bottom: 5px;">가입일</label>
            <p>{{ user.date_joined|date:"Y년 m월 d일 H:i" }}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; color: #7f8c8d; margin-bottom: 5px;">최근 로그인</label>
            <p>{{ user.last_login|date:"Y년 m월 d일 H:i" }}</p>
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; color: #7f8c8d; margin-bottom: 5px;">작성한 게시글 수</label>
            <p style="font-size: 1.1em; font-weight: bold;">{{ user.post_set.count }}개</p>
        </div>
    </div>

    <div style="margin-top: 20px; display: flex; gap: 10px;">
        <a href="{% url 'accounts:change_password' %}" style="background-color: #3498db; color: white; padding: 12px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
            비밀번호 변경
        </a>
        <a href="{% url 'board:list' %}" style="background-color: #95a5a6; color: white; padding: 12px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
            목록으로
        </a>
    </div>

    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
        <a href="{% url 'accounts:delete_account' %}" style="color: #e74c3c; text-decoration: none; font-size: 0.9em;">
            회원 탈퇴
        </a>
    </div>
</div>
{% endblock %}
```

## 8. 비밀번호 변경 템플릿

**accounts/templates/accounts/change_password.html**:

```html
<!-- accounts/templates/accounts/change_password.html -->

{% extends 'board/base.html' %}

{% block title %}비밀번호 변경 - Django 게시판{% endblock %}

{% block content %}
<div style="max-width: 500px; margin: 0 auto;">
    <h2 style="text-align: center; margin-bottom: 30px;">비밀번호 변경</h2>

    <form method="post">
        {% csrf_token %}

        {% if form.non_field_errors %}
            <div style="background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 15px; margin-bottom: 20px;">
                {{ form.non_field_errors }}
            </div>
        {% endif %}

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                현재 비밀번호 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.old_password }}
            {% if form.old_password.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.old_password.errors }}
                </div>
            {% endif %}
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                새 비밀번호 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.new_password1 }}
            {% if form.new_password1.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.new_password1.errors }}
                </div>
            {% endif %}
        </div>

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                새 비밀번호 확인 <span style="color: #e74c3c;">*</span>
            </label>
            {{ form.new_password2 }}
            {% if form.new_password2.errors %}
                <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                    {{ form.new_password2.errors }}
                </div>
            {% endif %}
        </div>

        <div style="display: flex; gap: 10px; margin-top: 30px;">
            <button type="submit" style="flex: 1; background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em;">
                변경하기
            </button>
            <a href="{% url 'accounts:profile' %}" style="flex: 1; background-color: #95a5a6; color: white; padding: 12px; text-decoration: none; border-radius: 4px; display: inline-block; text-align: center;">
                취소
            </a>
        </div>
    </form>
</div>

<style>
    input[type="password"] {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
    }

    input[type="password"]:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
</style>
{% endblock %}
```

## 9. 회원 탈퇴 템플릿

**accounts/templates/accounts/delete_account.html**:

```html
<!-- accounts/templates/accounts/delete_account.html -->

{% extends 'board/base.html' %}

{% block title %}회원 탈퇴 - Django 게시판{% endblock %}

{% block content %}
<div style="max-width: 500px; margin: 0 auto;">
    <h2 style="text-align: center; margin-bottom: 30px; color: #e74c3c;">회원 탈퇴</h2>

    <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 30px;">
        <p style="margin: 0; color: #856404;">
            <strong>⚠️ 주의사항</strong><br>
            • 회원 탈퇴 시 계정 정보는 즉시 삭제됩니다.<br>
            • 작성하신 게시글과 댓글도 모두 삭제됩니다.<br>
            • 삭제된 데이터는 복구할 수 없습니다.
        </p>
    </div>

    <form method="post">
        {% csrf_token %}

        <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 5px; font-weight: bold;">
                비밀번호 확인 <span style="color: #e74c3c;">*</span>
            </label>
            <input type="password" name="password" required
                   style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em;"
                   placeholder="비밀번호를 입력하세요">
            <small style="color: #7f8c8d;">본인 확인을 위해 비밀번호를 입력해주세요.</small>
        </div>

        <div style="display: flex; gap: 10px; margin-top: 30px;">
            <button type="submit"
                    onclick="return confirm('정말로 탈퇴하시겠습니까? 이 작업은 되돌릴 수 없습니다.');"
                    style="flex: 1; background-color: #e74c3c; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em;">
                탈퇴하기
            </button>
            <a href="{% url 'accounts:profile' %}"
               style="flex: 1; background-color: #95a5a6; color: white; padding: 12px; text-decoration: none; border-radius: 4px; display: inline-block; text-align: center;">
                취소
            </a>
        </div>
    </form>
</div>

<style>
    input[type="password"]:focus {
        outline: none;
        border-color: #e74c3c;
        box-shadow: 0 0 0 2px rgba(231, 76, 60, 0.2);
    }
</style>
{% endblock %}
```

## 10. base.html 네비게이션 업데이트

**board/templates/board/base.html**의 nav 부분을 수정하세요:

```html
<!-- board/templates/board/base.html -->

<nav>
    <ul>
        <li><a href="{% url 'board:list' %}">게시글 목록</a></li>
        {% if user.is_authenticated %}
            <li><a href="{% url 'board:create' %}">글쓰기</a></li>
            <li><a href="{% url 'accounts:profile' %}">{{ user.username }}님</a></li>
            <li><a href="{% url 'accounts:logout' %}">로그아웃</a></li>
        {% else %}
            <li><a href="{% url 'accounts:login' %}">로그인</a></li>
            <li><a href="{% url 'accounts:signup' %}">회원가입</a></li>
        {% endif %}
    </ul>
</nav>
```

### 설명

- `{% if user.is_authenticated %}`: 로그인 여부 확인
- 로그인된 사용자: 글쓰기, 프로필, 로그아웃 표시
- 로그인하지 않은 사용자: 로그인, 회원가입 표시

## 11. 게시글 상세 페이지 권한 설정

**board/templates/board/post_detail.html**에서 수정/삭제 버튼을 작성자만 볼 수 있도록 수정하세요:

```html
<!-- board/templates/board/post_detail.html -->

<footer style="border-top: 1px solid #ecf0f1; padding-top: 20px; margin-top: 30px;">
    <div style="display: flex; gap: 10px;">
        <a href="{% url 'board:list' %}" style="background-color: #95a5a6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
            목록으로
        </a>
        {% if user == post.author %}
            <a href="{% url 'board:update' post.pk %}" style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
                수정
            </a>
            <a href="{% url 'board:delete' post.pk %}" style="background-color: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;" onclick="return confirm('정말 삭제하시겠습니까?');">
                삭제
            </a>
        {% endif %}
    </div>
</footer>
```

## 12. 기능 테스트하기

### 1) 회원가입

1. 서버 실행: `python manage.py runserver`
2. 상단 메뉴에서 **"회원가입"** 클릭
3. 정보 입력 후 가입
4. 자동 로그인 확인

### 2) 로그아웃 및 로그인

1. 상단 메뉴에서 **"로그아웃"** 클릭
2. **"로그인"** 클릭하여 다시 로그인

### 3) 게시글 작성 (로그인 필수)

1. 로그아웃 상태에서 **"글쓰기"** 클릭
2. 로그인 페이지로 리다이렉트 확인
3. 로그인 후 자동으로 글쓰기 페이지로 이동

### 4) 프로필 확인

1. 상단 메뉴에서 본인 이름 클릭
2. 프로필 정보 확인

### 5) 비밀번호 변경

1. 프로필 페이지에서 **"비밀번호 변경"** 클릭
2. 현재 비밀번호와 새 비밀번호 입력
3. 변경 후에도 로그인 상태 유지 확인

### 6) 회원 탈퇴

1. 프로필 페이지 하단의 **"회원 탈퇴"** 링크 클릭
2. 비밀번호 입력 후 **"탈퇴하기"** 클릭
3. 확인 팝업에서 **"확인"** 클릭
4. 계정 삭제 및 메인 페이지로 이동 확인
5. 작성한 게시글도 함께 삭제되었는지 확인

## 📝 Part 5 정리

축하합니다! Part 5를 완료했습니다. 다음을 배웠습니다:

✅ 회원가입 기능 구현<br>
✅ 로그인/로그아웃 기능<br>
✅ 프로필 페이지<br>
✅ 비밀번호 변경<br>
✅ 회원 탈퇴 기능<br>
✅ 로그인 상태에 따른 UI 변경<br>
✅ 작성자 권한 확인<br>
✅ Django 비밀번호 검증 규칙 이해 및 커스터마이징

## 🔍 자주 발생하는 오류

### 1. 로그인 후 404 에러

**원인**: LOGIN_REDIRECT_URL 설정 오류

**해결**:
```python
# settings.py
LOGIN_REDIRECT_URL = '/'
```

### 2. 비밀번호가 너무 약하다는 에러

**원인**: Django의 기본 비밀번호 검증

**해결**: 더 강한 비밀번호 사용 (최소 8자, 숫자만 X, 흔한 비밀번호 X)

### 3. 회원가입 후 자동 로그인 안 됨

**원인**: `login(request, user)` 코드 누락

**해결**:
```python
user = form.save()
login(request, user)  # 이 줄 추가
```

### 4. 회원 탈퇴 시 비밀번호가 맞는데도 에러 발생

**원인**: 인증 함수 사용 오류

**해결**: views.py의 delete_account 함수에서 `authenticate()` 함수를 올바르게 사용했는지 확인

### 5. 비밀번호 검증을 비활성화했는데 여전히 검증이 작동함

**원인**: 서버를 재시작하지 않음

**해결**: `python manage.py runserver`를 재시작하여 settings.py 변경사항 반영

## 🔒 보안 권장사항

### 회원 탈퇴 시 고려사항

현재 튜토리얼에서는 회원 탈퇴 시 사용자와 관련된 모든 게시글이 삭제됩니다. 실제 서비스에서는 다음과 같은 옵션을 고려할 수 있습니다:

1. **즉시 삭제 (현재 방식)**
   - 장점: 개인정보 완전 삭제
   - 단점: 다른 사용자가 본 게시글도 사라짐

2. **작성자 익명화**
   ```python
   # 탈퇴 시 게시글 작성자를 익명으로 변경
   Post.objects.filter(author=user).update(author=None)
   user.delete()
   ```

3. **소프트 삭제 (비활성화)**
   - User 모델에 `is_active` 필드 사용
   - 계정만 비활성화하고 데이터는 유지

### 비밀번호 보안

운영 환경에서는 다음 설정을 권장합니다:

```python
# settings.py
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 10,  # 최소 10자 이상 권장
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
```

## 🚀 다음 단계

[Part 6: 댓글 기능](./Part6-Comments.md)에서 게시글에 댓글을 달 수 있는 기능을 만들어봅시다!

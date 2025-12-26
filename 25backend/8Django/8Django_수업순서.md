<span style="font-size:13px;">

=========================================

python 3.14 버전에서 진행함. 

Django==5.2.8

=========================================



0\.

pip install -r requirements.txt





1\.

django-admin startproject config .



2\.

python manage.py startapp blog



3\.

▶▶ config/setting.py 의 INSTALLED\_APPS = \[]  기재

=================

&nbsp;   # Third-party app (Django에서 제공해주는 것은 아님. 다른곳에서 제공..?)

&nbsp;   'rest\_framework',   # django를 RESTfull API로 확장. json기반

&nbsp;   'rest\_framework.authtoken', # 사용자에게 고정 토큰을 발급, api요청시 토큰으로 사용자 인증

&nbsp;   # DRF : (Django restfull framework) 표준 api 서버로 만들어주는 프레임워크

&nbsp;   'drf\_spectacular', # DRF API를 자동 문서화. Swagger UI, Redoc제공

&nbsp;   'django\_filters', # Query parameter 기반 필터링, DRF와 연동



&nbsp;   # local app

&nbsp;   'blog'

=================



▶▶ 하기 한국어 및 한국시간으로 변경

LANGUAGE\_CODE = 'ko-kr'

TIME\_ZONE = 'Asia/Seoul'







▶▶  INSTALLED\_APPS 아래쪽에 하기 내용 작성

=================

\# REST Framework 설정

REST\_FRAMEWORK = {

&nbsp;   'DEFAULT\_SCHEMA\_CLASS': 'drf\_spectacular.openapi.AutoSchema',

&nbsp;   'DEFAULT\_AUTHENTICATION\_CLASSES': \[

&nbsp;       'rest\_framework.authentication.TokenAuthentication',

&nbsp;       'rest\_framework.authentication.SessionAuthentication',

&nbsp;   ],

&nbsp;   'DEFAULT\_PERMISSION\_CLASSES': \[

&nbsp;       'rest\_framework.permissions.IsAuthenticatedOrReadOnly',

&nbsp;   ],

&nbsp;   'DEFAULT\_PAGINATION\_CLASS': 'rest\_framework.pagination.PageNumberPagination',

&nbsp;   'PAGE\_SIZE': 10,

&nbsp;   'DEFAULT\_FILTER\_BACKENDS': \[

&nbsp;       'django\_filters.rest\_framework.DjangoFilterBackend',

&nbsp;       'rest\_framework.filters.SearchFilter',

&nbsp;       'rest\_framework.filters.OrderingFilter',

&nbsp;   ],

}



\# drf-spectacular 설정

SPECTACULAR\_SETTINGS = {

&nbsp;   'TITLE': 'Blog API',

&nbsp;   'DESCRIPTION': 'Django REST Framework를 이용한 Blog API',

&nbsp;   'VERSION': '1.0.0',

&nbsp;   'SERVE\_INCLUDE\_SCHEMA': False,

}

=================





4\.

▶▶ config/urls.py 에서 path 추가 정의

=================

from django.urls import path, include

from drf\_spectacular.views import (

&nbsp;   SpectacularAPIView,

&nbsp;   SpectacularSwaggerView,

&nbsp;   SpectacularRedocView)





urlpatterns = \[

&nbsp;   path('admin/', admin.site.urls),

&nbsp;   path('api/', include('blog.urls')),

&nbsp;   # api 스키마 및 문서

&nbsp;   path('api/schema/', SpectacularAPIView.as\_view(), name='schema'),

&nbsp;   path('api/schema/swagger-ui/', SpectacularSwaggerView.as\_view(url\_name='schema'), name='swagger-ui'),

&nbsp;   path('api/schema/redoc/', SpectacularRedocView.as\_view(url\_name='schema'), name='redoc'),

]

=================





5\.

blog에 urls.py 생성 후 하기 내용 작성

=================

urlpatterns = \[

]

=================





6\.

python manage.py migrate





7\. 

슈퍼유저

python manage.py createsuperuser



8\.

python manage.py runserver 

page not found 나오는게 정상



config/settings.py 에 설정한 urlpatterns 하나씩 실행해보기

http://127.0.0.1:8000/api/schema/swagger-ui/

http://127.0.0.1:8000/api/schema/redoc/





9\. 

필요한 DB 만들기 : blog/models.py 파일 만들어서 DB 작성하기 

(카테고리, 태그, 포스트, 코멘트 등등)







10\.

blog/admin.py 작성



11\.

python manage.py makemigrations

python manage.py migrate





12\.

▶▶ 장고 테스트 데이터 생성하기

터미널: python manage.py shell ---------> 장고 시스템에 들어갈수 있음 

상기 터미널 엔터후에 하기 내용 복붙해서 터미널에 붙여넣기

(오류나면, python과 django 버전 확인하기)

pip install -U django==5.2

가상환경 python 버전은  3.14.2

=============================================

from django.contrib.auth.models import User

from blog.models import Category, Tag, Post, Comment

from django.utils import timezone



\# 사용자 생성

user = User.objects.create\_user(username='testuser', password='testpass123')



\# 카테고리 생성

django\_cat = Category.objects.create(name='Django', description='Django 관련 글')

python\_cat = Category.objects.create(name='Python', description='Python 관련 글')



\# 태그 생성

drf\_tag = Tag.objects.create(name='DRF')

api\_tag = Tag.objects.create(name='API')

rest\_tag = Tag.objects.create(name='REST')



\# 게시글 생성

post = Post.objects.create(

&nbsp;   title='Django REST Framework 시작하기',

&nbsp;   content='DRF를 사용하여 REST API를 만드는 방법',

&nbsp;   excerpt='DRF 입문 가이드',

&nbsp;   author=user,

&nbsp;   category=django\_cat,

&nbsp;   status='published',

&nbsp;   published\_at=timezone.now()

)

post.tags.add(drf\_tag, api\_tag, rest\_tag)



\# 댓글 생성

Comment.objects.create(

&nbsp;   post=post,

&nbsp;   author=user,

&nbsp;   content='좋은 글 감사합니다!'

)

=============================================

----> runserver 해서 admin에 접속해서 게시글이 작성된것을 확인





13\. blog/serializers.py 생성 및 작성

json 변환을 위해 필요





14\. blog/views.py 작성



15\. blog/ulsrs.py 에 상기 view 내용들 설정해주기



16\.  계정관리 (토큰인증)

python manage.py startapp accounts 새로운 앱 생성



17\. 로그인, 회원가입, 로그인처리

accounts/serializer.py 생성 및 작성



18\. 

accounts/views.py 작성



19\.

accounts/urls.py 생성 및 작성



20\.

&nbsp;blog/permissions.py  생성 및 작성



21\.

blog/views.py 수정하기.. 간단하게... (기존 기재사항은 주석처리됨)

class PostViewSet(viewsets.ModelViewSet):

class CommentViewSet(viewsets.ModelViewSet):



22\.

런서버 후

http://127.0.0.1:8000/api/auth/register/



http://127.0.0.1:8000/api/schema/swagger-ui/



상기 사이트에서 

/api/auth/login/

에서 try it out  --> 로그인 정보 작성 --> execute 하고나면 

--> 201 되면서 Response body에서

&nbsp; "token": "57b85890ccab1463afd2b06cc39c773f0b568420",

--> 이 토큰 값을 현재 사이트의 오른쪽 상단에 Authorize 를 클릭해서

--> tokenAuth에

Token 57b85890ccab1463afd2b06cc39c773f0b568420

이렇게 기재

--> /api/auth/me/  에서 try it out --> execute 하고나면

정상으로 하기와 같이 나오고

터미널은 "GET /api/auth/me/ HTTP/1.1" 200 88 이렇게 나옴



=================================

Curl



curl -X 'GET' \\

&nbsp; 'http://127.0.0.1:8000/api/auth/me/' \\

&nbsp; -H 'accept: application/json' \\

&nbsp; -H 'Authorization: Token 57b85890ccab1463afd2b06cc39c773f0b568420'

Request URL

http://127.0.0.1:8000/api/auth/me/

Server response

Code	Details

200	

Response body

Download

{

&nbsp; "id": 5,

&nbsp; "username": "testuser4",

&nbsp; "email": "test@sample.com",

&nbsp; "first\_name": "",

&nbsp; "last\_name": ""

}

Response headers

&nbsp;allow: GET,HEAD,OPTIONS 

&nbsp;content-length: 88 

&nbsp;content-type: application/json 

&nbsp;cross-origin-opener-policy: same-origin 

&nbsp;date: Fri,26 Dec 2025 07:09:17 GMT 

&nbsp;referrer-policy: same-origin 

&nbsp;server: WSGIServer/0.2 CPython/3.14.2 

&nbsp;vary: Accept 

&nbsp;x-content-type-options: nosniff 

&nbsp;x-frame-options: DENY 

=================================





http://127.0.0.1:8000/api/auth/me/ 

이렇게 직접 입력시 나오지 않는게 정상...



\[GPT]

왜 주소창에서는 토큰이 안 실리냐?

브라우저 주소창 요청은:

GET 요청 1개

헤더를 임의로 넣을 수 없음

Authorization 헤더 ❌

즉,🔐 Token 인증 API는 주소창으로 테스트하는 게 아님





\############################ 필터링 ##############################################3



23\. 필터링

blog/views.py에서 class PostViewSet 수정

http://127.0.0.1:8000/api/posts/?category=1 이렇게 필터가능하나, 좀더 .. 다듬..?



24\.

blog/filters.py 생성 및 작성






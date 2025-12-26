from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.models import User
from .serializers import UserSerializer, RegisterSerializer, LoginSerializer
from rest_framework.authentication import TokenAuthentication



class AuthViewSet(viewsets.GenericViewSet):
    '''인증관련 viewset'''
    serializer_class = UserSerializer
    
    def get_serializer_class(self):
        if self.action == 'register':
            return RegisterSerializer
        elif self.action == 'login':
            return LoginSerializer
        return UserSerializer
    

    @action(detail=False, methods=['POST'], permission_classes=[AllowAny]) # AllowAny 로그인없이 접근가능
    def register(self, request):
        '''회원가입'''
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            # instance가 없으면 create, 있으면 update
            user = serializer.save() # 내부 규칙에 의해서 update or create 자동 호출
            token,created = Token.objects.get_or_create( user = user) #user 같으면 get하고 다르면 create
            return Response({
                'token' : token.key,
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

    @action(detail=False, methods=['POST'], permission_classes =[AllowAny])
    def login(self,request):
        '''로그인'''
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data
            token,created = Token.objects.get_or_create( user = user) #user 같으면 get하고 다르면 create
            return Response({
                'token' : token.key,
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

    @action(detail=False, methods=['POST'], permission_classes =[IsAuthenticated]) #IsAuthenticated 인증되어있는 상태
    def logout(self,request):
        '''로그아웃'''
        request.user.auth_token.delete()
        return Response({'message':'로그아웃 되었습니다.'})
    
    @action(detail=False, methods=['GET'], permission_classes =[IsAuthenticated], authentication_classes=[TokenAuthentication]) #IsAuthenticated 인증되어있는 상태
    def me(self,request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)
    

    # 하기가 원본... 상기는 authentication_classes=[TokenAuthentication] 추가함
    # 원본
    # @action(detail=False, methods=['GET'], permission_classes =[IsAuthenticated]) #IsAuthenticated 인증되어있는 상태
        # 원본 상태로 하게되면,
            # DEFAULT_AUTHENTICATION_CLASSES = [SessionAuthentication]
            # 이 상태에서:
            # @action(permission_classes=[IsAuthenticated])
            # 👉 Swagger에서는 admin으로 인증됨
            # 👉 Token 넣어도 무시됨
            # 👉 api/auth/me가 계속 admin으로 나옴 😱


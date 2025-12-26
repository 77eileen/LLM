from rest_framework import permissions


class IsAuthorOrReadOnly(permissions.BasePermission):
    """
    작성자만 수정/삭제 가능, 나머지는 읽기만 가능
    """
    def has_object_permission(self, request, view, obj):
        # 읽기 권한은 모두 허용
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # 쓰기 권한은 작성자만
        return obj.author == request.user


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    객체의 소유자만 수정/삭제 가능
    """
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # owner 속성이 있으면 owner, 없으면 author 사용
        owner = getattr(obj, 'owner', None) or getattr(obj, 'author', None)
        return owner == request.user


class IsStaffOrReadOnly(permissions.BasePermission):
    """
    스태프만 생성/수정/삭제 가능
    """
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user and request.user.is_staff
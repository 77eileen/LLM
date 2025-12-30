import secrets
print(secrets.token_urlsafe(64))

    # JWT 토큰에 서명할 때 쓰는 key, 토큰이 서버에서 만든게 맞는지 검증 (실제는 .env에 저장해서 사용)
    # 서버를 재시작할때마다 secret_key 발급되면서 기존키는 무효화
    # 서버 실행시 기준키를 재발행 --> 모든 사용자의 토큰이 무효과 --> 강제 로그아웃
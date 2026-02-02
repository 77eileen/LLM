EC2 접속 (Windows 사용자 필독)
SSH 키 권한 문제(`WARNING: UNPROTECTED PRIVATE KEY FILE!`)를 해결하기 위해 **PowerShell**에서 아래 명령어를 **딱 한 번** 실행하세요.

```powershell
# 1. 상속 권한 제거
icacls "C:\AWS\my-key-2026-2.pem" /inheritance:r
# 2. 현재 사용자에게만 읽기 권한 부여
icacls "C:\AWS\my-key-2026-2.pem" /grant:r "$($env:USERNAME):(R)"
```

이제 접속합니다:
```powershell
ssh -i "C:\AWS\my-key-2026-2.pem" ec2-user@<내-EC2-퍼블릭-IP>
```

위의 방법중에 에러가 발생하면 
```
   # BUILTIN\Users 그룹 제거
   icacls "C:\AWS\my-key-2026-01.pem" /remove "BUILTIN\Users"
   
   # Authenticated Users 그룹도 있다면 제거 (선택사항)
   icacls "C:\AWS\my-key-2026-01.pem" /remove "NT AUTHORITY\Authenticated Users"

   # 접속 
   ssh -i "C:\AWS\my-key-2026-2.pem" ec2-user@<내-EC2-퍼블릭-IP>
```

### 3-2. 서버 환경 세팅 (한 번만 실행)
```bash
# 업데이트 및 DB 클라이언트 설치
sudo dnf update -y
sudo dnf install mariadb105 -y

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 라이브러리 설치
pip install fastapi uvicorn sqlalchemy pymysql pydantic cryptography
```

### 3-3. 코드 배포 (복사 붙여넣기)
가장 간단한 방법은 `nano` 에디터를 쓰는 것입니다.

1.  EC2 터미널에서 `nano main.py` 입력.
2.  로컬에서 작성한(그리고 RDS 정보가 입력된) `main.py` 내용을 전체 복사(Ctrl+A, Ctrl+C).
3.  터미널에 마우스 우클릭으로 붙여넣기.
4.  `Ctrl + O` (저장) -> `Enter` -> `Ctrl + X` (종료).

### 3-4. 서버 실행
```bash
# 백그라운드 실행 아님 (터미널 끄면 꺼짐)
uvicorn main:app --host 0.0.0.0 --port 8000
```
이제 `http://<EC2-IP>:8000/docs`에 접속하면 전 세계 어디서든 내 API를 사용할 수 있습니다!

---

### [Tip] 서버를 계속 켜두고 싶다면? (nohup)
터미널을 꺼도 서버가 죽지 않게 하려면:
```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
```
*   종료하려면: `pkill uvicorn`

### 사용 중인 프로세스 확인 및 종료
```
  1. 8000번 포트를 누가 쓰고 있는지 확인 (PID 확인)
  sudo lsof -i :8000

  2. 확인된 PID(숫자)를 강제 종료
  sudo kill -9 <PID>
  예: sudo kill -9 12345

   1
   2 *(만약 `lsof` 명령어가 없다고 나오면 `sudo yum install lsof` 로 설치하세요)*
   3
   4 ### 2. 한 번에 강제 종료 (가장 빠름)
   5 귀찮으시면 아래 명령어로 8000번 포트를 쓰는 프로그램을 즉시 죽일 수 있습니다.
  sudo fuser -k 8000/tcp
   1
   2 ### 3. 서버 재실행
   3 종료 후 다시 원래 명령어를 실행하시면 됩니다.
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```
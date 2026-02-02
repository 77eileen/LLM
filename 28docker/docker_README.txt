
2026.01.07  docker로 배포하기

1. 임의 폴더에
app, docker, upload 폴더 만들기
.env, docker-compose.yml, requirements.txt 파일 만들기

2. 
docker-compose.yml 작성

3.
가상환경 실행하고 pip install -r requirements.txt 실행

4. app 폴더에서
api, core, db, model, schema, services 폴더 만들기

- 패키지 만들기
app 폴더 및 상기 6개 폴더 전부에 __init__.py 만들기

5. app/core 폴더
config.py 파일 생성 및 작성

6. docker폴더
app, nginx 폴더 만들기

7. docker/app
Dockerfile 생성 및 작성

8. app 폴더에
main.py 생성 및 작성

9. docker/nginx
nginx.conf 파일 생성 및 작성

10.
.env 작성

11.
docker desktop 실행 후 창내리고,
(터미널: 실행)docker-compose up -d --build
 에러가 나도 터미널에서는 안보이고 콘솔창에 보임..

- 상기 설치 끝나고 나서, 다른 터미널에서
docker-compose logs -f
실행하면, 로그 확인 가능함

12.
docker desktop에서
임의의 메인폴더 이름(fastapi-mysql~)으로 컨테이너가 만들어진것을 확인!
컨테이너 화살표 누르면 db, app, nginx 도 만들어진것을 확인

13.
localhost 접속
localhost/docs 접속해서 try it - excute해서 제대로 나오는거 확인해보기

14. mysql 접속해보기
+ 클릭
standard(TCP/IP)
Username: 상기 .env와 동일하게 
Password: 상기 .env와 동일하게

15.
docker 멈추고 싶으면
터미널에서
docker-compose down


16. app/db폴더
session.py 생성 및 작성

17. app/models 폴더
base.py 생성 및 작성
item.py 생성 및 작성 ==> CRUD를 위해 테이블을 만듬
file_model.py 생성 및 작성 ==> 

18. app/db폴더
init_db.py 생성 및 작성

19. app/schemas 폴더
item.py 생성 및 작성 ==> schemas의 데이터를 검정


20. app폴더
main.py 파일 수정

21.
(실행)docker-compose up -d --build
(로그확인 및 app로그저장?) docker-compose logs -f app 
-- mysql에서 db에서 테이블에 2개가 생긴것을 확인

(종료)  docker-compose down


22. app/services 폴더
item_service.py 생성 및 작성
file_service.py 생성 및 작성


23.  라우팅 설정
app/api폴더
v1/endpoint 폴더 만들기
v1폴더에 api.py 생성 및 작성

v1/endpoint 폴더에서
files.py, items.py 생성 및 작성

app/models 폴더에
file_model.py 생성 및 작성

app/schemas 폴더에
file.py, item.py 생성 및 작성


24. docker의 컨테이너 진입 (docker는 리눅스 환경)
docker exec -it <컨테이너명> bash
docker exec -it fastapi_app bash ==> 컨테이너 fastapi_app안으로 들어감
pwd  
exit

docker exec -it fastapi_db bash ==> 컨테이너 fastapi_db안으로 들어감 (mysql안으로 들어온것임)
ls  ==> 안에 있는 파일 다 보기


======================= 배포
25. docker 계정 생성

26.
배포용 이미지 새로 빌드
docker build -t fastapi-server:v1 -f docker/app/Dockerfile .

27.
Docker Hub 형식으로 태그 변경 (YOUR_ID를 본인 아이디로 변경!)
docker tag fastapi-server:v1 YOUR_DOCKER_ID/fastapi-server:v1
docker tag fastapi-server:v1 77eileen/fastapi-server:v1

28.
버전 지정 푸시
docker push YOUR_DOCKER_ID/fastapi-server:v1
docker push 77eileen/fastapi-server:v1

docker hub로 들어가면 생성된 것을 확인할 수 있음

29.
배포할 새로운 폴더 만들고
그안에 docker-compose.yml 과 .env 만들기
docker폴더/nginx폴더 만들고 nginx.conf 파일 생성 및 작성
(실행)docker-compose up -d --build

30.
배포할 새로운 폴더명을 "docker backend project"로 수정
해당 폴더 압축하기
(docker에서 기존에 사용한 cotainer,image 등등 모두 삭제하기
docker관련 폴더도 모두 삭제)
압축파일을 배포.
압축파일을 풀고.. cmd 
해당 폴더로 cd 경로 이동
docker-compose up -d
하면 컨테이너가 만들어지면서 재생됨



=======================
디스크 용량 관리

1️⃣ 프로젝트 컨테이너/볼륨/이미지 삭제
docker compose -f compose.yaml down --rmi all -v


2️⃣ Docker 전체 정리 (모든 프로젝트)
docker system prune -a --volumes
**삭제되는 것:**
- ✅ 중지된 모든 컨테이너
- ✅ 사용 안 하는 모든 이미지
- ✅ 사용 안 하는 모든 볼륨
- ✅ 사용 안 하는 모든 네트워크

docker system prune -a --volumes -f
-f: 확인 없이 강제 삭제

========================

vmmem 메모리 제한 걸어주기 위해서.. 

[파워셀]
notepad $env:USERPROFILE\.wslconfig

메모장에 입력
[wsl2]
memory=4GB
processors=2
swap=0
localhostForwarding=true
메모장 저장 -> 메모장 닫기

wsl --shutdown


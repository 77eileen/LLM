### 도커를 이용한 ubuntu 설치
- docker 공식사이트에서 자신의 운영체재에 맞는 버전 설치(window :  ..... AMD64)
- powershell을 관리자 권한으로 열어서
- wsl --install
- 재부팅
- docker desktop 실행해서 docker가 실행상태
- powershell에서 docker info 정상 실행 확인
- docker run -it ubuntu bash  실행해서 컨터이너 안의 ubuntu에 접속한다


### 1. 디렉토리 이동 및 위치 확인
- `pwd`: 현재 위치한 디렉토리의 경로를 출력합니다.
- `ls`: 현재 디렉토리의 파일과 폴더 목록을 보여줍니다.
- `ls -al`: 숨겨진 파일을 포함하여 상세 정보를 출력합니다.
- `cd /`: 최상위(root) 디렉토리로 이동합니다.
- `cd ~`: 현재 사용자의 홈 디렉토리로 이동합니다.

### 2. 파일 및 디렉토리 관리
- `mkdir my_folder`: 'my_folder'라는 이름의 새 디렉토리를 만듭니다.
- `touch hello.txt`: 내용이 비어있는 'hello.txt' 파일을 생성합니다.
- `echo "Hello Linux" > hello.txt`: 파일에 텍스트를 입력합니다.
- `cat hello.txt`: 파일의 내용을 터미널에 출력합니다.
- `cp hello.txt copy.txt`: 파일을 복사합니다.
- `mv copy.txt my_folder/`: 파일을 폴더 안으로 이동시킵니다.
- `rm hello.txt`: 파일을 삭제합니다.
- `rm -rf my_folder`: 디렉토리와 그 안의 내용을 모두 강제로 삭제합니다.

### 3. 리눅스 디렉토리 구조(FHS)
```
윈도우와 달리 C D 드라이브 개념이 없고 모든 것인 루트(/)에서 시작하는 단일 구조
```
- /(Root) : 최상위
- /bin & /usr/bin : 기본적인 명령어들(ls, cp)이 위치한 곳
- /etc : 시스템 설정 파일들의 보관소(윈도우의 제어판 + 레지스트리 역활)
- /home : 일반사용자들의 개인 디렉토리
- /var : 로그파일, 데이터베이스 등 수시로 변화는 파일 저장소
- /tmp : 임시 파일 저장소, 재부팅시 자동삭제가 될수도 있는 파일들

### 4. 목록 보기
- cd /etc
- ls                      # 그냥 이름만 보기
- ls -l                   # 상세 정보 보기 (권한, 소유자, 크기, 날짜)
- ls -a                   # 숨겨진 파일(.으로 시작)까지 모두 보기
- ls -al                  # 숨겨진 파일을 포함해 상세 정보 보기
- ls -lh                  # 파일 크기를 사람이 읽기 편하게(K, M, G 단위) 표시
- ls -R                   # 하위 디렉토리 내용까지 펼쳐서 보기 (재귀적, 스크롤 주의!)
- ls -d */                # 파일은 제외하고 디렉토리만 골라서 보기

### 5. 디렉토리 생성
- cd ~
- mkdir project                   # 폴더 하나 생성
- mkdir -p project/backend/api    # 하위 폴더까지 한 번에 생성 (-p: parents)
- mkdir project/frontend project/db # 여러 폴더 동시에 생성
- ls -R project                   # 구조 확인

### 6. 파일생성
- cd project
- touch README.md                 # 빈 파일 생성
- touch index.html style.css      # 여러 파일 동시 생성
- touch .gitignore                # 숨김 파일 생성
- ls                              # .gitignore 안 보임
- ls -a                           # 이제 보임

### 7. 복사의 다양한 옵션
- cp README.md README_bak.md      # 파일 복사
- cp -r backend backend_backup    # 폴더 통째로 복사할 땐 -r (recursive) 필수!
- cp -v index.html index_v2.html  # 복사 과정을 자세히 출력 (-v: verbose)

### 8. 이동(mv)와 이름변경
- mv style.css frontend/
- mv index_v2.html old_index.html # 이름 변경 (같은 폴더 내 이동 = 이름 변경)

### 9. 삭제(rm) - 주의
- rm old_index.html               # 파일 삭제 (되살릴 수 없음!)
- rm -r backend_backup            # 폴더 삭제는 -r 필수
- rm -rf project/db               # 묻지도 따지지도 않고 강제 삭제 (-f: force, 매우 위험)
# *주의: rm -rf / 명령은 시스템 전체를 날릴 수 있습니다. 절대 금지!*

```
my_work라는 폴더를 만들고 그 안에 year_2024, year_2025 폴더를 만드세요.
year_2024 안에 report1.txt, report2.txt, image.jpg 파일을 만드세요.
와일드카드(*)를 사용하여 year_2024의 모든 .txt 파일을 year_2025로 복사하세요.
year_2024 폴더를 통째로 archive라는 이름으로 변경하세요.
archive 폴더 안에 있는 image.jpg를 삭제하세요.
```

### 10 도커 컨터이너 확인 및 실행
- docker run -it ubuntu bash  # 도커를 이용해서 ubuntu를 컨테이너에 넣고 실행하고 컨테이너에 진입
- docker ps -a  # 현재 실행 및 종료된 컨테이너 목록
- 만약에 중지되어 있으면 docker desktop으로 실행 또는 docker start [CONTAINER ID]
- 실행중인 컨테이너 안에 진입 : docker exec -it [CONTAINER ID] bash

### 11 에디터
- apt update && apt install nano -y   # 패키지 목록 갱신 후 nano 설치
- ls -R /etc > file_list.txt      # /etc 목록을 파일로 저장 (내용이 꽤 김)
- cat file_list.txt               # 순식간에 지나가서 앞부분을 못 봄
- head -n 5 file_list.txt         # 딱 첫 5줄만 보기
- tail -n 5 file_list.txt         # 딱 마지막 5줄만 보기

```
nano 에디터
```
- 입력 : "ServerIP = 127.0.0.1" 입력
- 복사/붙여넣기 : Alt+6( 복사 )  Ctrl+U (붙여넣기)
- 잘라내기 : Ctrl+k (한줄 삭제/ 잘라내기)
- 검색 : Ctrl+w  누르고 찾을 단어 입력
- 저장 및 종료 : Ctrl+o(Write out) -> Enter -> Ctrl+x(Exit)

### 12. 로그 실시간 감시
- touch access.log
- tail -f access.log      # 파일에 내용이 추가될 때마다 실시간으로 화면에 출력

```
실습:
Nano를 사용하여 diary.txt를 만드세요.
내용에 "Today is Monday"를 적고, 그 줄을 복사해서 5번 붙여넣으세요.
3번째 줄의 "Monday"를 "Friday"로 수정하세요.
저장 후 빠져나와 cat으로 내용을 확인하세요.
head 명령어를 써서 이 파일의 앞 3줄만 출력해보세요.
```

### 13. Grep & Find
``` 
샘플 데이터
echo -e "Apple\nBanana\nCherry\nDate\nElderberry" > fruits.txt
echo -e "Error: 404\nInfo: Login success\nError: 500\nWarn: Low disk" > server.log
```
- grep "Apple" fruits.txt         # 기본 검색
- grep -i "apple" fruits.txt      # 대소문자 무시 (-i)
- grep -v "Error" server.log      # "Error"가 *없는* 줄만 출력 (-v: 반전)
- grep -n "Error" server.log      # 몇 번째 줄인지 줄 번호 표시 (-n)
- grep -c "Error" server.log      # 매칭되는 줄의 개수 카운트 (-c)
- grep "^Error" server.log        # "Error"로 *시작하는* 줄만 검색 (정규식 ^)

```
find 명령어
```
- find /etc -name "*.conf"        # /etc 아래 확장자가 .conf인 모든 파일 찾기
- find /usr -size +10M            # 10MB보다 큰 파일 찾기
- find . -type d                  # 현재 경로 아래의 '디렉토리'만 찾기
- find . -name "fruits.txt" -delete # 찾아서 바로 삭제 (주의!)

```
파이프 응용
```
# /etc 디렉토리 파일 중 'conf'가 들어가는 파일 개수 세기
- ls /etc | grep "conf" | wc -l

# 프로세스 목록 중 'bash'만 찾아서 보기
- ps aux | grep bash

```
/var 디렉토리 전체에서 이름에 log가 들어가는 파일을 찾으세요.
위에서 찾은 파일 중, 파일 크기가 1kbyte 이상인 것만 찾아보세요. (find /var -name "*log*" -size +1k)
ls -al /etc 명령어의 결과에서 "root"라는 단어가 몇 번 나오는지 파이프와 grep -c를 이용해 한 줄 명령어로 구하세요.
```

```
cat << EOF > grep_practice.txt
Apple Pie
Banana Split
Cherry Cake
apple juice
Banana Bread
Date 2024
Error: File not found
Error: Access denied
Warning: Disk full
Info: Update completed
EOF
```

| 옵션 | 설명 | 기억법 |
| :--- | :--- | :--- |
| `-i` | 대소문자 무시 | **I**gnore case |
| `-v` | 패턴 제외 (반전) | In**V**ert |
| `-n` | 줄 번호 표시 | **N**umber |
| `-r` | 하위 폴더 포함 검색 | **R**ecursive |
| `-c` | 개수 카운트 | **C**ount |
| `-l` | 파일명만 출력 | **L**ist file |
| `^` | 라인의 시작 | (Shift+6) |
| `$` | 라인의 끝 | (Shift+4) |

### 14. 권한
```
# 실습용 파일 생성
touch secret.txt
ls -l secret.txt
```
| 구분 | 권한(User) | 권한(Group) | 권한(Others) | 링크수 | 소유자 | 그룹 | 크기 | 날짜 | 이름 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `-` | `rw-` | `r--` | `r--` | `1` | `root` | `root` | `0` | ... | `secret.txt` |
*(맨 앞이 `d`면 디렉토리, `-`면 파일입니다)*

2진수 비트 계산(r=4, w=2, x=1)을 사용합니다.

| 권한 조합 | 숫자 | 의미 |
|:---:|:---:|:---|
| `rwx` | 4+2+1 = **7** | 읽고 쓰고 실행 가능 (관리자 권한) |
| `rw-` | 4+2+0 = **6** | 읽고 쓰기 가능 (일반 파일 표준) |
| `r-x` | 4+0+1 = **5** | 읽고 실행 가능 (일반 프로그램/폴더 표준) |
| `r--` | 4+0+0 = **4** | 읽기만 가능 (중요 설정 파일) |
| `---` | 0+0+0 = **0** | 아무것도 못함 (접근 금지) |

#### 권한변경 - 숫자모드
```
chmod 777 secret.txt  # rwxrwxrwx (누구나 다 가능 - 보안 위험!)
ls -l secret.txt

chmod 755 secret.txt  # rwxr-xr-x (주인은 맘대로, 남들은 실행/읽기만)
ls -l secret.txt

chmod 600 secret.txt  # rw------- (나만 읽고 쓰기 - 개인키/비번 파일 필수)
ls -l secret.txt
```

#### 권한변경 - 심볼릭모드
```
chmod u+x secret.txt  # User에게 eXecute 권한 추가
chmod g-w secret.txt  # Group에게 Write 권한 제거
chmod o=r secret.txt  # Others는 Read만 가능하게 고정
chmod a-r secret.txt  # All(모두)에게서 Read 권한 뺏기 (나도 못 읽음!)
```

```
# 실습용 사용자 추가
useradd -m developer

# 파일 소유자를 developer로 변경
chown developer secret.txt
ls -l secret.txt       # 소유자가 root -> developer로 바뀜

# 소유자와 그룹을 동시에 변경 (사용자:그룹)
chown root:root secret.txt
```

```
mkdir -p /var/www/html
touch /var/www/html/index.html

# 디렉토리는 755 (누구나 들어올 수 있음)
chmod 755 /var/www/html

# 파일은 644 (누구나 읽을 수 있음)
chmod 644 /var/www/html/index.html

# 확인
ls -ld /var/www/html
ls -l /var/www/html/index.html
```

### 15. 시스템 관리자 와 패키지
패키지 관리 라이브러리 라이프사이클
```
apt update                      # 서버에서 최신 패키지 정보를 가져옴
apt install htop curl tree -y   # htop, curl, tree 설치
apt remove tree                 # tree 삭제 (설정 파일은 남음)
apt purge tree                  # tree 완전 삭제 (설정 파일 포함)
apt autoremove                  # 쓰지 않는 의존성 패키지 자동 정리
```
시스템 상태 확인
```
top -b -n 1                     # 현재 시스템 상태 스냅샷 (CPU, 메모리)
htop                            # 컬러풀한 그래픽 모드 (F10 또는 q로 종료)
df -h                           # 디스크 용량 확인 (Disk Free) -h: Human readable
du -sh /var/log                 # 특정 디렉토리의 용량 확인 (Disk Usage)
free -h                         # 남은 메모리(RAM) 확인
```
프로제스 제어
```
# 백그라운드에서 오래 도는 프로세스 시뮬레이션
sleep 1000 &                    # 1000초 동안 대기하는 명령을 백그라운드(&) 실행
ps aux | grep sleep             # sleep 프로세스 찾기 (PID 확인)

# 프로세스 강제 종료 (PID가 1234라면)
# kill 1234
# kill -9 1234                  # 말을 안 들으면 강제 종료 (-9)
pkill sleep                     # 이름으로 바로 종료
```

```
실습:
ncdu라는 디스크 사용량 분석 도구를 apt로 설치해보세요.
설치된 ncdu를 실행하여 현재 디렉토리(/)의 용량 분포를 확인하고 종료(q)하세요.
현재 실행 중인 프로세스 중 메모리를 가장 많이 사용하는 프로세스 상위 5개를 출력하는 명령어를 찾아보세요. 
(힌트: ps 명령어의 정렬 옵션 또는 top 사용)
```

### 16. 쉘 스크립트 프로그래밍(Shell Scripting)
```
Shebang (#!/bin/bash): 스크립트 첫 줄에 필수. 어떤 쉘로 실행할지 지정.
```

nano hello.sh
```
#!/bin/bash
NAME="LinuxUser"              # 변수 지정 (공백 없어야 함)
echo "Hello, $NAME!"
echo "오늘 날짜는 $(date) 입니다." # 명령어 결과 넣기 $()

echo -n "당신의 이름은? "
read USER_INPUT               # 사용자 입력 받기
echo "반갑습니다, $USER_INPUT 님."
```
chmod +x hello.sh

./hello.sh

#### 조건문(check_file.sh)
```
#!/bin/bash
FILE='mylog.txt'
if [ -f "$FILE" ]; then
        echo "$FILE is exist"
        cat $FILE
else
        echo "$FILE is not exist, and create that"
        touch $FILE
fi

chmod +x check_file.sh
./check_file.sh
```
#### 순환문
```
#!/bin/bash
for i in {1..5}
do
        echo "work file $i created...."
        touch "job_$i.data"
        sleep 0.5
done
ls *.data
```
#### 시스템 모니터링
```
#!/bin/bash

# Server Name
server_name=$(hostname)

# Color variables for better readability (Industry standard practice for CLI tools)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CLEAR='\033[0m'

# Helper function for colored output
function print_color() {
    local color=$1
    local text=$2
    echo -e "${color}${text}${CLEAR}"
}

# --- Monitor Functions ---

function memory_check() {
    echo ""
    print_color "$BLUE" "Memory usage on ${server_name}:"
    free -h
    echo ""
}

function cpu_check() {
    echo ""
    print_color "$BLUE" "CPU load on ${server_name}:"
    uptime
    echo ""
}

function disk_check() {
    echo ""
    print_color "$BLUE" "Disk usage on ${server_name}:"
    df -h | grep -vE '^Filesystem|tmpfs|cdrom' # Exclude tmpfs and cdrom for cleaner output
    echo ""
}

function tcp_check() {
    echo ""
    print_color "$BLUE" "TCP connections count:"
    # In production, simple connection counting is common for quick health checks
    cat /proc/net/tcp | wc -l
    echo ""
}

function kernel_check() {
    echo ""
    print_color "$BLUE" "Kernel version:"
    uname -r
    echo ""
}

# --- Interactive Menu ---

function show_menu() {
    echo -e "\n${GREEN}--- System Monitor Menu ---${CLEAR}"
    echo "1) Check Memory Usage"
    echo "2) Check CPU Load"
    echo "3) Check Disk Usage"
    echo "4) Check TCP Connections"
    echo "5) Check Kernel Version"
    echo "6) Run All Checks"
    echo "0) Exit"
    echo -n "Choose an option: "
}

function main() {
    while true; do
        show_menu
        read -r option
        case $option in
            1) memory_check ;;
            2) cpu_check ;;
            3) disk_check ;;
            4) tcp_check ;;
            5) kernel_check ;;
            6) 
                memory_check
                cpu_check
                disk_check
                tcp_check
                kernel_check
                ;;
            0) 
                print_color "$GREEN" "Exiting..."
                exit 0 
                ;;
            *) 
                print_color "$RED" "Invalid option. Please try again."
                ;;
        esac
        
        echo -n "Press Enter to continue..."
        read -r
    done
}

# Start the script
main

```

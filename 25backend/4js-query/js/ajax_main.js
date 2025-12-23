// JavaScript (언어)
//  └─ jQuery (라이브러리)
//      └─ 우리가 쓰는 $(), .on(), .ajax()

// api를 테스트하기 위해서 통신이 필요하고 Mock 서버인 json server로 실습
// json 서버용 directory 를 만들고 해당 directory로 이동해서
// db.json 파일을 생성하고 --DB
// https://nodejs.org/ko/download 설치
// 터미널에서 실행 : npm install -g json-server
// 터미널에서 실행 : json-server --watch db.json --port 3000


// 화면이 로드 될때까지 기다렸다가 완료되면 js 실행
$(document).ready(function(){
    // 1. 데이터 로드
    loadUsers();

    //======================================================================
    // ================== 2. 원래 있던 element에 대한 이벤트 등록===========
    //======================================================================
    // CREATE 행 추가  prompt
    $("#addBtn").on('click',function(){
        const name = prompt('이름 입력');
        const email = prompt('이메일 입력');
        if(!name || !email) return;        
        const user = {name, email}
        createUser(user);
    });
   
   
    // 업데이트(update)
    $("#userTable").on('click','.edit',function(){
        const name = prompt('수정할 이름');
        const email = prompt('수정할 이메일');
        const id = $(this).closest('tr').data('id');
        data = {'name':name, 'email':email}
        updateUser(id,data)
    });    
    
    
    // 삭제 : 단일 행   테이블의 데이터는 동적으로 생성했기때문에 이벤트를 직접 발생시키지 못하고 위임해야 한다
    $("#userTable").on('click','.remove',function(){
        const id = $(this).closest('tr').data('id')   // 태그 안에 있는 어트리뷰트(attr) data-id        
        deleteUser(id)
    });


    // 삭제 버튼(selected checkbox)  다중 삭제
    const deleteRequests = [] //ajax 요청을 저장
    $("#deleteBtn").on('click',function(){
        $('.chk:checked').each(function(){
            const id = $(this).closest('tr').data('id')
            deleteRequests.push(
                $.ajax({
                    url:`http://localhost:3000/users/${id}`,
                    method:'DELETE'
                })
            )
        });

    // 모든 삭제 요청이 끝날 때까지 기다림
    $.when(...deleteRequests).then(function(){
        alert('선택된 항목 삭제 완료')
        loadUsers()
    }).fail(function(){
        alert('일부 삭제 중 오류가 발생')
        loadUsers()
    });
   });
});





// 함수 등록
function deleteUser(id){
    $.ajax({
        url:`http://localhost:3000/users/${id}`,
        method : 'DELETE',
        success:function(){
            alert('삭제되었습니다.');
            loadUsers(); // 목록 갱신
        }
    });
}


function updateUser(id,data){
    $.ajax({
        url:`http://localhost:3000/users/${id}`,
        method : 'PUT',
        contentType : 'application/json',
        data : JSON.stringify(data),
        success:function(){
            alert('수정되었습니다.');
            loadUsers(); // 목록 갱신
        }
    });
}

function loadUsers(){
    $.ajax({
        url:'http://localhost:3000/users',
        method: 'GET',
        success:function(users){
            $('#userTable').empty();
            users.forEach(user => {
                $('#userTable').append(
                    `
                    <tr data-id="${user.id}">
                        <td><input type="checkbox" class="chk"></td>
                        <td>${user.id}</td>
                        <td>${user.name}</td>
                        <td>${user.email}</td>
                        <td>
                            <button class="edit">Modify</button>
                            <button class="remove">remove</button>
                        </td>
                    </tr>
                    `
                )
            });
        },
        error:function(){
            alert('목록조회 실패');
        }
    });
}

// POST 사용자 추가
function createUser(data){
    $.ajax({
        url:'http://localhost:3000/users',
        method: 'POST',
        contentType : 'application/json',
        data : JSON.stringify(data),
        success:function(){
            alert('등록되었습니다.');
            loadUsers(); // 목록 갱신
        }
    });
}
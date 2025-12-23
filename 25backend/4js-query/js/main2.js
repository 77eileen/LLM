
//브라우저 개발자도구 콘솔에 메시지 출력
// 👉 JS 파일이 HTML에 제대로 연결됐는지 확인용
console.log('js 연결확인') 


// dom - html 전체 구조를 객체화 한것
$(document).ready(
    function(){
        // 초기 렌더링. 리스트업(ready)
        let users = [
            {id:1, name:'다정', email:'dj@test.com'},
            {id:2, name:'지은', email:'je@test.com'},
            {id:3, name:'소영', email:'sy@test.com'},
        ]
        // for user in users:
        // user
        function renderTable(){
            $('#userTable').empty();

                users.forEach(user=>{
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
    }
        
        renderTable();

        // 전체 선택 해제
        $("#checkall").on('change', function(){
            $('.chk').prop('checked', this.checked)
        });

        // 동적생성한 요소는 이벤트 위임으로 부모에게 이벤트를 위임해서 처리
        $('#userTable').on('change', '.chk', function(){
            $("#checkall").prop("checked",
                $(".chk").length==$(".chk:checked").length
            )
        });

        // CREATE 행 추가 : prompt
        $("#addBtn").on('click', function(){
            const name=prompt('이름 입력');
            const email=prompt('이메일 입력');
            if (!name||!email) return;
            const newId = users.length? users[users.length-1].id+1 : 1; // javascript or java or C++ or C# 모두 동일하게 사용
            // check = age >= 19 ? '성인':'미성년'; // 3항 연산자
            // 상기 문장은 하기 내용과 동일함
            // if (users.length){
            //     const newId=users[users.length-1].id+1
            // }
            // else{
            //     const newId=1
            // }
        users.push({id:newId, name, email})
        renderTable();
        });
        // 삭제 : 단일행 테이블의 데이터는 동적으로 생성했기 때문에 이벤트를 직접 발생시키지 못하고 위임해야 한다
        $("#userTable").on('click', '.remove', function(){
            const id = $(this).closest('tr').data('id')    // 태그 안에 있는 attribute(attr)를 data-id로 표현?
            users = users.filter(u => u.id != id)
            renderTable()
        });
        // 다중 선택 삭제 ()
        $("#deleteBtn").on('click',function(){
            const ids = []
            $('.chk:checked').each(function(){
                ids.push( $(this).closest('tr').data('id'))
            });
            users = users.filter( u => !ids.includes(u.id)) // 다중삭제 (단일삭제는 users.filter(u => u.id != id);)
            renderTable()
        });
        // 업데이트 (update) -- 하기 코드 실제 html에 없음
        $('#editBtn').on('click', '.edit', function(){
            const name = prompt ('수정할 이름');
            const email= prompt ('수정할 이메일');
            const idx = $(this).closest('tr').data('id')-1;
            const user = users[idx];
            user.name = name;
            user.email = email;
            renderTable();

        });
    }
);

// 페이지가 완전히 로드되면 내부 코드를 실행
$(document).ready(function(){ 
    // 데이터 로드, 1. 페이지가 열리자마다 할일 목록을 불러옴
    read_todos();
    // 2. 등록 버튼이 있는 폼(#todoForm)에 이벤트 연결
    // id가 todoForm인 폼에서 submit 이벤트(등록 버튼 클릭 등)가 발생하면 지정된 함수를 실행
    $('#todoForm').on('submit',function(e){
        // 폼 제출시 페이지가 새로고침 되는 기본 동작을 막음
        e.preventDefault(); 
        // 할일을 추가하는 함수를 호출
        addTodo();        
    });
});

// 3. '할일 추가' 기능 함수
function addTodo(){
    // 입력된 제목과 설명을 가져옴
    const title = $('#todoTitle').val().trim();
    const description = $('#todoDescription').val().trim();
    // 서버(main.py)로 보낼 데이터 객체를 생성
    const todoData = {
        title:title,
        description:description || null  //디스크립션이 있으면 디스크립션 없으면 널을 가져온다
    }
    //AJAX를 사용해 서버에 POST 요청을 보냄
    $.ajax({
        url :  '/api/todos',   // 요청을 보낼 서버 주소    
        method : 'POST', // 직접 추가하니까 post 방식  // HTTP 요청방식
        data :JSON.stringify(todoData),             // 실제 보낼 데이터
        contentType:'application/json',             // 보내는 데이터 형식은 JSON이라고 명시
        success:function(newTodo){                  // 요청 성공 시 실행될 함수 / newTodo는 백엔드단에 있는 함수
            console.log('추가 성공', newTodo)
            read_todos();  // 추가한 목록을 갱신
        },
        error:function(error){                      // 요청 실패 시 실행될 함수
            console.log('추가 실패', error)
        }
    });
}
// 4.'할일 목록 조회' 기능 함수
function read_todos(){
    //ajax를 사용해 서버에 GET 요청을 보냄
    $.ajax({
        url:'http://localhost:8000/api/todos',
        method:'GET', //조회니까 get 방식
        success:function(todos){
            const $todolists = $('#todoList')  //태그를 객체로 가져올때 변수명 앞에 $붙인다
            $todolists.empty();     // 비워준다. 하나가 추가되면 전부 지우고 새로 추가한다. 화면상태와 서버상태를 동일하게 맞추기 위해
            // 서버에서 받은 모든 할일(todos)에 대해 반복
            todos.forEach(function(todo){
                // 각 할일에 대한 HTML 코드를 생성
                const $todoItem = `
                <div class="todo-item" data-id="${todo.id}">
                    <h3>${todo.title}</h3>
                    <p>${todo.description}</p>
                    <p>${todo.completed}</p>
                    <p>${todo.created_at}</p>
                </div>
                `
                // 생성된 HTML을 #todoList 영역에 추가
                $todolists.append($todoItem)
            });


        },
        error:function(error){
            console.log('읽기 실패', error);
        }
    })
}
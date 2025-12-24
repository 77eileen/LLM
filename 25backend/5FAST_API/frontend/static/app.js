// Step 5: TODO List - jQuery로 FastAPI와 통신

// API 기본 URL
const API_URL = 'http://localhost:8000/api';

// 현재 필터 상태
let currentFilter = 'all';

// ============================================
// 페이지 로드시 실행
// ============================================
$(document).ready(function() {
    console.log('TODO App 시작');
    
    // 초기 데이터 로드
    loadTodos();
    loadStats();
    
    // 이벤트 리스너 등록
    setupEventListeners();
});

// ============================================
// 이벤트 리스너 설정
// ============================================
function setupEventListeners() {
    // TODO 추가 폼 제출
    $('#todoForm').on('submit', function(e) {
        e.preventDefault();
        addTodo();
    });
    
    // 필터 버튼 클릭
    $('.btn-filter').on('click', function() {
        const filter = $(this).data('filter');
        changeFilter(filter);
    });
}

// ============================================
// TODO 목록 불러오기
// ============================================
function loadTodos() {
    console.log('TODO 목록 로딩...');
    
    $.ajax({
        url: `${API_URL}/todos`,
        method: 'GET',
        success: function(todos) {
            console.log(`${todos.length}개의 TODO 로드 성공`);
            displayTodos(todos);
        },
        error: function(error) {
            console.error('TODO 로드 실패:', error);
            alert('TODO 목록을 불러오는데 실패했습니다.');
        }
    });
}

// ============================================
// TODO 화면에 표시
// ============================================
function displayTodos(todos) {
    const $todoList = $('#todoList');
    const $emptyState = $('#emptyState');
    
    // 필터링
    const filteredTodos = filterTodos(todos);
    
    // 리스트 비우기
    $todoList.empty();
    
    // TODO가 없으면 빈 상태 표시
    if (filteredTodos.length === 0) {
        $emptyState.show();
        return;
    }
    
    $emptyState.hide();
    
    // 각 TODO 아이템 생성
    filteredTodos.forEach(function(todo) {
        const $todoItem = createTodoElement(todo);
        $todoList.append($todoItem);
    });
}

// ============================================
// TODO 아이템 HTML 생성
// ============================================
function createTodoElement(todo) {
    const completedClass = todo.completed ? 'completed' : '';
    const checkedAttr = todo.completed ? 'checked' : '';
    const description = todo.description ? 
        `<div class="todo-description">${escapeHtml(todo.description)}</div>` : '';
    
    const html = `
        <div class="todo-item ${completedClass}" data-id="${todo.id}">
            <div class="todo-header">
                <input 
                    type="checkbox" 
                    class="todo-checkbox" 
                    ${checkedAttr}
                    onchange="toggleTodo(${todo.id}, this.checked)"
                >
                <div class="todo-title">${escapeHtml(todo.title)}</div>
            </div>
            ${description}
            <div class="todo-meta">
                <span class="todo-date">${todo.created_at}</span>
                <div class="todo-actions">
                    <button class="btn btn-danger" onclick="deleteTodo(${todo.id})">
                        삭제
                    </button>
                </div>
            </div>
        </div>
    `;
    
    return $(html);
}

// ============================================
// TODO 추가
// ============================================
function addTodo() {
    const title = $('#todoTitle').val().trim();
    const description = $('#todoDescription').val().trim();
    
    if (!title) {
        alert('할일 제목을 입력하세요.');
        return;
    }
    
    const todoData = {
        title: title,
        description: description || null
    };
    
    console.log('TODO 추가 중...', todoData);
    
    $.ajax({
        url: `${API_URL}/todos`,
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(todoData),
        success: function(newTodo) {
            console.log('TODO 추가 성공:', newTodo);
            
            // 폼 초기화
            $('#todoForm')[0].reset();
            
            // 목록 새로고침
            loadTodos();
            loadStats();
            
            showMessage('TODO가 추가되었습니다.');
        },
        error: function(error) {
            console.error('TODO 추가 실패:', error);
            alert('TODO 추가에 실패했습니다.');
        }
    });
}

// ============================================
// TODO 완료 상태 토글
// ============================================
function toggleTodo(todoId, completed) {
    console.log(`TODO ${todoId} 상태 변경: ${completed}`);
    
    $.ajax({
        url: `${API_URL}/todos/${todoId}`,
        method: 'PUT',
        contentType: 'application/json',
        data: JSON.stringify({ completed: completed }),
        success: function(updatedTodo) {
            console.log('TODO 업데이트 성공:', updatedTodo);
            
            loadTodos();
            loadStats();
        },
        error: function(error) {
            console.error('TODO 업데이트 실패:', error);
            alert('상태 변경에 실패했습니다.');
            
            loadTodos();
        }
    });
}

// ============================================
// TODO 삭제
// ============================================
function deleteTodo(todoId) {
    if (!confirm('정말 삭제하시겠습니까?')) {
        return;
    }
    
    console.log(`TODO ${todoId} 삭제 중...`);
    
    $.ajax({
        url: `${API_URL}/todos/${todoId}`,
        method: 'DELETE',
        success: function() {
            console.log('TODO 삭제 성공');
            
            loadTodos();
            loadStats();
            
            showMessage('TODO가 삭제되었습니다.');
        },
        error: function(error) {
            console.error('TODO 삭제 실패:', error);
            alert('TODO 삭제에 실패했습니다.');
        }
    });
}

// ============================================
// 통계 불러오기
// ============================================
function loadStats() {
    $.ajax({
        url: `${API_URL}/health`,
        method: 'GET',
        success: function(stats) {
            $('#total-count').text(stats.total);
            $('#completed-count').text(stats.completed);
            $('#pending-count').text(stats.pending);
        },
        error: function(error) {
            console.error('통계 로드 실패:', error);
        }
    });
}

// ============================================
// 필터 변경
// ============================================
function changeFilter(filter) {
    currentFilter = filter;
    
    $('.btn-filter').removeClass('active');
    $(`.btn-filter[data-filter="${filter}"]`).addClass('active');
    
    loadTodos();
}

// ============================================
// TODO 필터링
// ============================================
function filterTodos(todos) {
    if (currentFilter === 'all') {
        return todos;
    } else if (currentFilter === 'completed') {
        return todos.filter(todo => todo.completed);
    } else if (currentFilter === 'pending') {
        return todos.filter(todo => !todo.completed);
    }
    return todos;
}

// ============================================
// 유틸리티 함수
// ============================================

// HTML 이스케이프 (XSS 방지)
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 메시지 표시 (선택사항)
function showMessage(message) {
    console.log(message);
}

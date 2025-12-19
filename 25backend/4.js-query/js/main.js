
//브라우저 개발자도구 콘솔에 메시지 출력
// 👉 JS 파일이 HTML에 제대로 연결됐는지 확인용
console.log('js 연결확인') 


// dom - html 전체 구조를 객체화 한것
$(document).ready(
    function(){
        console.log('jquery 준비완료');
        // HTML 문서가 전부 로딩된 후에 실행하라는 의미
        // JS가 HTML보다 먼저 실행되면 요소를 못 찾을 수 있어서 필요함

    
        $("#btn").click(function(){
            $('#text').text('버튼 클릭됨');
        })

        // 전체선택
        $("#checkall").on('change', function(){
            $('.chk').prop('checked', this.checked);
            const checked = $('.chk:checked').length
            // 개수를 카운트 (전체선택했을 때)
            $('#count').text(checked)
        });


        // 개별체크로 전체선택 컨트롤
        $('.chk').on('change', function(){
            const total = $('.chk').length
            const checked = $('.chk:checked').length
            $('#checkall').prop('checked', total==checked)
            // 개수를 카운트
            $('#count').text(checked)
        });

        // 선택 삭제
        $('#deleteBtn').click(function(){
            $('.tchk:checked').each(function(){
                $(this).closest('tr').remove()
            });
        });

        // 버튼 비활성화(중복 클릭 방지) 저장 결제 api 호출
        $('#saveBtn').click(function(){
            $(this).prop('disabled',true)

            setTimeout( ()=>{
                $(this).prop('disabled',false)
            },2000  );

        });

        //입력값을 실시간 검증
        $('#username').on('input', function(){  // "()=>" 함수는 this를 바인딩 하지 않고 window객체를.. "fuction()"으로 변경 또는 "(e)=>"" 이렇게 쓰기
            const val = $(this).val()
            if (val.length < 3){
                $('#msg').text('3자 이상 입력').css('color', 'red')
            }
            else{
                $('#msg').text('사용가능').css('color', 'blue')
            }
        });

        //동적 요소 추가
        let cnt =1 ;
        $('#addBtn').click(function(){  // "()=>" 함수는 this를 바인딩 하지 않고 window객체를.. "fuction()"으로 변경 또는 "(e)=>"" 이렇게 쓰기
            $('#list').append(
                `
                <div class='item'>
                    동적항목 ${cnt++}  
                    <span class='remove'>삭제</span>
                </div> 
                `
            );  // cnt++ ----> cnt +=1 과 같은 의미, `` ---> f'' 과 같은 기능  
        })
// 이벤트 위임
// 이벤트 위임은 on 메서드로 사용
// 부모요소.on (이벤트 종류, 자식요소 선택자, 실행함수)  $('#list').on('click', '.remove', function()
// 이유는 javascript로 동적으로 생성한 element는 javascript에서 인식하지 못함
// javascript보다 나중에 생성된 element이기 때문에
// html은 이벤트 버블링이 있어서 모든 이벤트는 부모로 전파된다는 속성을 이용해서 역으로 
// 해당 부모가 이벤트를 감지해서 발생하면 자식의 바로 부모엘리먼트를 제거할 수 있음
// 이것이 바로 이벤트 위임
// 테이블, 리스트, 댓글

        $('#list').on('click', '.remove', function(){
            $(this).parent().remove();
        })

    }
);



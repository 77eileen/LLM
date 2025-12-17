# 여기 파일에서 가져온것
# C:\00AI\LLM\23Agent\1Basic_Agent\1_2_multiple_agent.ipynb

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json

# 에이전트 상태
class AgentState(Enum):
    IDLE = 'idle'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    ERROR = 'error'


# 데이터클래스
@dataclass
class Message:
    '''에이전트 간 메세지'''
    message_id : str
    sender_id : str
    receiver_id : Optional[str]
    content : Dict[str, Any]
    timestamp: str
    def to_dict(self) -> Dict[str,Any]:
        return{
            'id': self.message_id,
            'sender': self.sender_id,
            'receiver' : self.receiver_id,
            'content' : self.content,
            'timestamp' : self.timestamp
        }
    
class SpecializedAgent :
    '''특화된 에이전트'''
    def __init__ (self, name:str, speciality:str):
        '''
        Args:
            name : 에이전트 이름
            speciality : 전문분야
        '''
        self.agent_id = str(uuid.uuid4())[:8]
        self.name = name
        self.speciality = speciality
        self._state = AgentState.IDLE
        self._inbox : List[Message] = []
        self._outbox : List[Message] = []
    
    def receive_message(self, message: Message):
        '''메세지 수신'''
        self._inbox.append(message)
    
    def send_message(self, receiver_id:str, content:Dict[str, Any]):
        '''메세지 전송'''
        message=Message(
            message_id=str(uuid.uuid4())[:8],
            sender_id = self.agent_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self._outbox.append(message)
        return message
    
    def process_inbox(self) -> list[Dict[str,Any]]:
        '''받은 메세지 처리'''
        self.set_state(AgentState.PROCESSING) # self._state = self.set_state(AgentState.PROCESSING) 이렇게 기재하면, set_state return 값이 없으므로..? 받으면 안됨..?? 무슨말
        results = []
        for message in self._inbox:
            result = self._handle_message(message)
            results.append(result)
        self._inbox = [] # 상기 처리된 메세지를 제거함
        self.set_state(AgentState.COMPLETED)
        return results
    
    def _handle_message(self, message:Message) -> Dict[str, Any]:
        '''메세지 처리(오버라이드 가능 (다시 구현 가능?))'''
        return{
            'status':'handled',
            'message_id': message.message_id,
            'content': message.content
        }
    
    def get_state(self) -> str: 
        return self._state.value
        # get_state: 상태를 "읽기 전용 문자열"로 제공 / 값반환이므로 매개변수 필요없음/ return값음 있음
        # get_state: 상태가 궁금해? 보기좋은 문자열만 보여줄게
    def set_state(self, state:AgentState):  
        self._state = state
        # set_state: 상태를 "정해진 규칙(Enum)"으로만 변경 / 매개변수는 있으나, return값이 없음.
        # set_state: 상태를 바꾸고 싶어? AgentState 중 하나만 가져와
    def get_info(self) -> Dict[str, Any]:
        '''에이전트의 상태를 반환'''
        return{
            'id': self.agent_id,
            'name': self.name,
            'speciality': self.speciality,
            'state': self.get_state(),
            'inbox_size': len(self._inbox),
            'outbox_size' : len(self._outbox)
        }
        

# 사용자가 입력이 들어오면, 서비스를 분기함 (ex 지도, 검색 등---> 라우팅
# 라우터 클래스 : 강사님이 주신 것으로 원래 파일과 다름!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class Coordinator:
    def __init__(self):
        self.agents: Dict[str, SpecializedAgent] = {}
    
    def register_agent(self, agent: SpecializedAgent):
        self.agents[agent.agent_id] = agent
    
    def route_message(self):
        for agent in self.agents.values():
            for message in agent._outbox:
                if message.receiver_id in self.agents:
                    receiver = self.agents[message.receiver_id]
                    receiver.receive_message(message)
                    print(f'  ✓ {message.message_id}: {agent.name} → {receiver.name}')
            agent._outbox = []
    
    def process_all_agents(self):
        for agent in self.agents.values():
            if agent._inbox:
                agent.process_inbox()


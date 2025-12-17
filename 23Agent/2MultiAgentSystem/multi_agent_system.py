# Multi-Agent + RAG + Knowledge Graph
# 영화 질문에 답하기 위해
# 벡터 검색 Agent와 그래프 검색 Agent를 따로 두고,
# 필요에 따라 쓰게 만든 Multi-Agent RAG 시스템



# 사용자 질문
#    ↓
# Coordinator (중앙 관제)
#    ↓
#  ┌───────────────┬─────────────────────┐
#  │ VectorDBAgent │ KnowledgeGraphAgent │
#  │ (의미 기반)   │ (관계 기반)         │
#  └───────────────┴─────────────────────┘
#    ↓
# 결과 종합 → 답변

# 📌 중요
# Agent 하나 = 역할 하나
# 검색 방식이 다르면 Agent도 분리

# 왜 Agent를 두개로 나눴을까?
# ❌ 하나의 Agent로 다 하면?
    # 의미 검색도 해야 함
    # 관계 탐색도 해야 함
    # 코드 복잡
    # 역할 불명확
# ✅ Agent 분리하면?
#        Agent                  전문분야
#  ──────────────────────────────────────────
#      VectorDBAgent          (의미 기반)       
#  ──────────────────────────────────────────
#   KnowledgeGraphAgent       (관계 기반)         


# SpecializedAgent / Message 구조의 읨
    # message = Message(
    #     sender="Coordinator",
    #     receiver="VectorDBAgent",
    #     content={"query": "..."}
    # )
# 📌 Agent는 서로 직접 함수 호출 안 함
# 📌 메시지로 소통
# 👉 이게 바로 Agent 시스템




# SAMPLE_MOVIES 구조에서 하기 부분 검색가능
    # plot : 벡터검색
    # director/actors : 그래프관계
    # genre : 카테고리 연결
    # rating : 메타데이터 필터

SAMPLE_MOVIES = [
    {
        "id": "m1",
        "title": "The Shawshank Redemption",
        "year": 1994,
        "director": "Frank Darabont",
        "genre": ["Drama", "Crime"],
        "actors": ["Tim Robbins", "Morgan Freeman"],
        "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "rating": 9.3
    },
    {
        "id": "m2",
        "title": "The Godfather",
        "year": 1972,
        "director": "Francis Ford Coppola",
        "genre": ["Crime", "Drama"],
        "actors": ["Marlon Brando", "Al Pacino"],
        "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "rating": 9.2
    },
    {
        "id": "m3",
        "title": "The Dark Knight",
        "year": 2008,
        "director": "Christopher Nolan",
        "genre": ["Action", "Crime", "Drama"],
        "actors": ["Christian Bale", "Heath Ledger"],
        "plot": "When the menace known as the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests.",
        "rating": 9.0
    },
    {
        "id": "m4",
        "title": "Pulp Fiction",
        "year": 1994,
        "director": "Quentin Tarantino",
        "genre": ["Crime", "Drama"],
        "actors": ["John Travolta", "Samuel L. Jackson"],
        "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
        "rating": 8.9
    },
    {
        "id": "m5",
        "title": "Inception",
        "year": 2010,
        "director": "Christopher Nolan",
        "genre": ["Action", "Sci-Fi", "Thriller"],
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
        "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        "rating": 8.8
    }
]


from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid, json, os
import chromadb 
from chromadb.config import Settings
import openai
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# 기본 구조
from defaultAgent import AgentState, Message, SpecializedAgent, Coordinator


# RAG에 특화된 Agent 
# VectorDBAgent : 의미로 찾는 Agent (질문 의미랑 가장 비슷한 영화(semantic search) 찾아줘) 
class VectorDBAgent (SpecializedAgent):
    '''벡터 DB 검색 에이전트'''
    def __init__(self, name:str):
        super().__init__(name, 'vector_search')
        # chromaDB 초기화
        self.client = chromadb.Client(Settings(
            anonymized_telemetry = False,   # False 익명의 사용정보를 수집 및 전송을 비활성 (보안/사내에서 적합. 폐쇄적인곳에서는 False)
            allow_reset = True # 외부에서 db를 reset 할 수 있는 api 허용
        ))
        try:
            self.client.delete_collection('movies')
        except:
            pass
        self.collection = self.client.create_collection(
            name = 'movies',
            metadata= {'description' : 'movie information database'}
        )
        self._initialize_db()
    def _get_embedding(self, text:str) -> List[float]:
        '''openai 임베딩 생성'''
        try:
            response = openai.embeddings.create(
                model = 'text-embedding-3-small',
                input = text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f'임베딩 생성 실패 : {e}')
            return [0.0]*1536
    def _initialize_db(self):
        '''영화 데이터를 vectorDB에 저장'''
        for movie in SAMPLE_MOVIES:
            doc_text = f"{movie['title']} ({movie['year']}). {movie['plot']}"

            self.collection.add(
                ids=[movie['id']],
                documents=[doc_text],
                embeddings=[self._get_embedding(doc_text)],
                metadatas=[{
                    'title':movie['title'],
                    'year':movie['year'],
                    'director':movie['director'],
                    'rating':movie['rating'],
                    'genre': ','.join(g for g in movie['genre'])
                }]
            )

    def _handle_message(self, message:Message)->Dict[str,Any]:
        content = message.content
        query = content.get('query', '')
        top_k = content.get('top_k', 3)
        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        retrieved_docs =[]
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                retrieved_docs.append({
                    'id' : results['ids'][0][i],
                    'content':doc,
                    'metadata':metadata,
                    'similarity' : 1-distance
                })
        return {
            'status':'retrieved',
            'query' : query,
            'results' : retrieved_docs,
            'count' : len(retrieved_docs)
        }
    


# pip install networkx
# KnowledgeGraphAgent : 관계로 찾는 Agent
# ---> 누가 누구랑 연결돼 있지? 의미는 없고, 관계만 있음
# [Movie] ──directed_by──> [Director]
#    │
#    ├──stars──> [Actor]
#    │
#    └──genre──> [Genre]

# NetworkX = “그래프 자료구조를 파이썬으로 다루는 라이브러리”


import networkx as nx
class KnowledgeGraphAgent(SpecializedAgent):
    '''지식 그래프 검색 Agent (Networkx 기반)'''
    def __init__(self, name:str):
        super().__init__(name, 'knowledge_graph')
        self.graph = nx.DiGraph()    # “이 Agent가 관리하는 ‘관계 지도’ 하나 만들었다”
        self._initialize_graph()

    def _initialize_graph(self):
        '''영화 관계 그래프 생성'''
        for movie in SAMPLE_MOVIES:
            # 영화 노드
            self.graph.add_node(movie['id'], type='movie', title=movie['title'], year=movie['year'])
            # 감독 노드 및 관계
            director_id = f"idr_{movie['director'].replace(' ', '_')}"
            self.graph.add_node(director_id, type='direct', name=movie['director'])
            self.graph.add_edge(movie['id'], director_id,relation='directed_by')
            # 배우 노드 및 관계
            for actor in movie['actors']:
                actor_id = f"act_{actor.replace(' ', '_')}"
                self.graph.add_node(actor_id, type='actor', name=actor)
                self.graph.add_edge(movie['id'], actor_id,relation='stars')
            # 장르 노드 및 관계
            for genre in movie['genre']:
                genre_id = f"gen_{genre}"
                self.graph.add_node(genre_id, type='genre', name=genre)
                self.graph.add_edge(movie['id'], genre_id, relation='genre')
            # 레이팅 노드  및 관계
            rating_id = f"rating_{movie['rating']}"
            self.graph.add_node(rating_id, type='rating', name=movie['rating'])
            self.graph.add_edge(movie['id'], rating_id,relation='rating')


    def _handle_message(self, message: Message) -> Dict[str, Any]:
        content = message.content
        query_type = content.get('query_type', 'related')
        entity_id = content.get('entity_id', '')
        
        print(f"\n[{self.name}] 그래프 검색: {query_type} for {entity_id}")
        
        if query_type == 'related':
            # 관련 엔티티 찾기
            if entity_id in self.graph:
                neighbors = list(self.graph.neighbors(entity_id))
                related = []
                
                for neighbor in neighbors:
                    node_data = self.graph.nodes[neighbor]
                    edge_data = self.graph.edges[entity_id, neighbor]
                    related.append({
                        'id': neighbor,
                        'type': node_data.get('type', ''),
                        'name': node_data.get('name', node_data.get('title', '')),
                        'relation': edge_data.get('relation', '')
                    })
                
                print(f"  ✓ {len(related)}개 관련 엔티티 발견")
                
                return {
                    'status': 'found',
                    'entity_id': entity_id,
                    'related': related,
                    'count': len(related)
                }
        
        elif query_type == 'find_by_director':
            director_name = content.get('director', '')
            director_id = f"dir_{director_name.replace(' ', '_')}"
            
            if director_id in self.graph:
                # 감독의 영화 찾기
                movies = [n for n in self.graph.predecessors(director_id) 
                         if self.graph.nodes[n]['type'] == 'movie']
                
                movie_list = []
                for movie_id in movies:
                    movie_data = self.graph.nodes[movie_id]
                    movie_list.append({
                        'id': movie_id,
                        'title': movie_data['title'],
                        'year': movie_data['year']
                    })
                
                print(f"  ✓ {len(movie_list)}개 영화 발견")
                
                return {
                    'status': 'found',
                    'director': director_name,
                    'movies': movie_list,
                    'count': len(movie_list)
                }
        
        return {'status': 'not_found'}


class LLMAgent(SpecializedAgent):
    '''LLM 기반의 응답 생성 Agent'''
    def __init__(self, name:str):
        super().__init__(name, 'LLM generation')
    def _handle_message(self, message:Message) -> Dict[str, Any]:
        content = message.content
        query = content.get('query', '')
        context = content.get('context', '')
        # context 정리
        # context_text = '\n'.join([ item.get('content', item) for item in context])
        context_text = '\n'.join([ item for item in context]) # 문자열이라 get안됨
        prompt = f''' 다음 정보를 바탕으로 사용자 질문에 답변해주세요
        context :
        {context_text}

        
        질문: {query}
        답변은 한국어로 작성하고, 제공된 정보만 사용하여 정확하게 답변하세요
        '''

        try:
            response = openai.chat.completions.create(model='gpt-4o-mini', 
                                           messages=[
                                               {'role': 'system', 'content':'당신은 영화정보 전문가입니다.'},
                                               {'role':'user', 'content':prompt}
                                           ],
                                           temperature=0,  # 환각방지를 위해 0으로해봄
                                           max_tokens=500)
            answer = response.choices[0].message.content
            return {
                'status':'generated',
                'query':query,
                'answer':answer,
                'model': 'gpt-4o-mini'
            }
        except Exception as e:
            print(f"LLM 생성 실패 {e}")
            return {
                'status':'error',
                'error':str(e)
            }


# RAG 파이프라인 오케스트레이터????
class OrchestratorAgent(SpecializedAgent):
    """RAG 파이프라인 Orchestrator"""
    
    def __init__(self, name: str, coordinator: Coordinator):
        super().__init__(name, "orchestration")
        self.coordinator = coordinator
        self.vector_agent_id = None
        self.graph_agent_id = None
        self.llm_agent_id = None
    
    def set_agents(self, vector_id: str, graph_id: str, llm_id: str):
        self.vector_agent_id = vector_id
        self.graph_agent_id = graph_id
        self.llm_agent_id = llm_id
    
    def _handle_message(self, message: Message) -> Dict[str, Any]:
        content = message.content
        query = content.get('query', '')
        
        print(f"\n{'='*70}")
        print(f"[{self.name}] RAG 파이프라인 시작: '{query}'")
        print(f"{'='*70}")
        
        # Step 1: Vector DB 검색
        self.send_message(self.vector_agent_id, {
            'query': query,
            'top_k': 3
        })
        self.coordinator.route_message()
        results = self.coordinator.agents[self.vector_agent_id].process_inbox()
        
        vector_results = results[0]['results'] if results else []
        
        # Step 2: Knowledge Graph 검색 (첫 번째 결과 기반)
        graph_results = []
        if vector_results:
            movie_id = vector_results[0]['id']
            self.send_message(self.graph_agent_id, {
                'query_type': 'related',
                'entity_id': movie_id
            })
            self.coordinator.route_message()
            graph_res = self.coordinator.agents[self.graph_agent_id].process_inbox()
            if graph_res:
                graph_results = graph_res[0].get('related', [])
        
        # Step 3: 컨텍스트 통합
        context = []
        for item in vector_results:
            context.append(item['content'])
        
        # for item in graph_results[:6]:
        for item in graph_results:
            context.append(f"{item['name']} ({item['type']})")
        
        # Step 4: LLM 응답 생성
        self.send_message(self.llm_agent_id, {
            'query': query,
            'context': context
        })
        self.coordinator.route_message()
        llm_results = self.coordinator.agents[self.llm_agent_id].process_inbox()
        
        final_answer = llm_results[0].get('answer', '') if llm_results else ''
        
        print(f"\n{'='*70}")
        print(f"최종 답변:\n{final_answer}")
        print(f"{'='*70}\n")
        
        return {
            'status': 'completed',
            'query': query,
            'answer': final_answer,
            'sources': {
                'vector_results': len(vector_results),
                'graph_results': len(graph_results)
            }
        }



def run_rag_system():
    # 코디네이터 생성
    coordinator = Coordinator()
    # 에이전트 생성 및 등록
    # 에이전트 생성 및 등록
    vector_agent = VectorDBAgent("VectorDB-Agent")
    graph_agent = KnowledgeGraphAgent("KnowledgeGraph-Agent")
    llm_agent = LLMAgent("LLM-Agent")
    orchestrator = OrchestratorAgent("Orchestrator", coordinator)
    
    coordinator.register_agent(vector_agent)
    coordinator.register_agent(graph_agent)
    coordinator.register_agent(llm_agent)
    coordinator.register_agent(orchestrator)
    
    orchestrator.set_agents(
        vector_agent.agent_id,
        graph_agent.agent_id,
        llm_agent.agent_id
    )
    
    # 테스트 쿼리들
    test_queries = [
        # "감옥에서 벌어지는 이야기를 다룬 영화 추천해줘",
        # "Christopher Nolan 감독의 영화는?",
        "범죄와 드라마 장르의 영화 중 평점이 높은 것은?"
    ]
    
    print("\n" + "="*70)
    print("테스트 쿼리 실행")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n Query {i}: {query}")
        
        # 오케스트레이터에게 쿼리 전송
        orchestrator.send_message(
            orchestrator.agent_id,
            {'query': query}
        )
        
        # 메시지 라우팅 및 처리
        coordinator.route_message()
        orchestrator.process_inbox()
        
        input("\n계속하려면 Enter를 누르세요...")
    
    print("\n" + "="*70)
    print("시스템 종료")
    print("="*70)


if __name__ == '__main__':
    run_rag_system()
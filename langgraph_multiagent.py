import os
import json
import fitz
from glob import glob
from langchain_ollama import ChatOllama
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from langchain_core.tools import Tool

# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.tools import SerperSearchRun

global_embed_model = SentenceTransformer('BAAI/bge-m3')
# global_embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')

# 상태 정의
class AgentState(TypedDict):
    query: str
    conversation_context: Optional[str]
    search_response: Optional[str]
    conversation_summary: Optional[str]
    final_answer: Optional[str]

class SearchAgent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchAgent, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.llm = ChatOllama(model="gemma3:12b-it-q4_K_M")
        # self.llm = ChatOllama(model='qwen3:14b-q4_K_M')
        # self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embed_model = global_embed_model
        self.search = DuckDuckGoSearchRun()
        
        print("PDF 로딩 및 청크 생성 중...")
        raw_docs = self.extract_texts_from_pdfs("./data")
        self.chunks = self.split_text_to_chunks(raw_docs)
        embeddings = self.embed_model.encode([doc.page_content for doc in self.chunks])
        self.faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.faiss_index.add(np.array(embeddings))
        self.chunk_store = self.chunks
        self.initialized = True

        self.tools = [
            Tool(name="LocalSearch", func=self.faiss_search, description="FAISS 기반 로컬 문서 검색"),
            Tool(name="WebSearch", func=self.web_search_tool_func, description="DuckDuckGo 기반 웹 검색")
        ]

    def extract_texts_from_pdfs(self, folder_path="./data"):
        pdf_paths = glob(os.path.join(folder_path, "*.pdf"))
        # pdf_paths = glob(os.path.join(folder_path, "*.txt"))
        pdf_paths.sort()
        all_texts = []
        for pdf_path in pdf_paths:
            print(pdf_path)
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    all_texts.append(Document(page_content=text.strip()))
        return all_texts

    def split_text_to_chunks(self, docs, chunk_size=1000, chunk_overlap=100):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[".", "\n"]
        )
        return splitter.split_documents(docs)

    def faiss_search(self, query: str, threshold: float = 0.6, k: int = 7) -> str:
        q_emb = self.embed_model.encode([query])
        D, I = self.faiss_index.search(np.array(q_emb), k=k)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.chunk_store):
                continue
            # L2 거리를 유사도 점수로 변환 (0~1, 작을수록 유사)
            similarity = 1 / (1 + dist)  # 간단한 정규화 방식
            if similarity < threshold:
                continue  # 신뢰도 점수가 임계값 미만이면 제외
            result = f"문서 {idx} (신뢰도: {similarity:.2f}):\n{self.chunk_store[idx].page_content}"
            results.append(result)
        
        if not results:
            print("Observation: No relevant documents found in local DB with sufficient confidence.")
            return f"로컬 DB에서 신뢰도 {threshold:.2f} 이상의 관련 문서를 찾지 못했습니다."
        
        print(f"Observation: Found {len(results)} relevant documents with confidence >= {threshold:.2f}.")
        return "\n\n".join(results)

    def web_search_tool_func(self, query: str) -> str:
        print("Action: Performing web search...")
        try:
            result = self.search.invoke(query)
            print(f"Observation: Web search result: {result[:100]}...")
            return result
        except Exception as e:
            print(f"Observation: Web search failed: {e}")
            return f"웹 검색 실패: {e}"
            
    def process_query(self, state: AgentState) -> AgentState:
        query = state["query"]
        conversation_context = state.get("conversation_context", "")
        
        print(f"Thought: Processing query: {query}")
        
        local_result = self.faiss_search(query, threshold=0.57)  # 신뢰도 임계값 추가
        if "관련 문서를 찾지 못했습니다" in local_result:
            web_result = self.web_search_tool_func(query) # 로컬 결과가 없을 경우에만 웹 검색
        else:
            web_result = "웹 검색 생략 (로컬에서 충분한 정보 확보)"

        prompt = f"""
        당신은 당뇨병 관리에 특화된 AI입니다. 다음 정보를 바탕으로 질문에 **한글로만**, 목록 형식으로 간결히 답변하세요.
        **질문**: {query}
        **로컬 검색 결과**: {local_result[:500]}...
        **웹 검색 결과**: {web_result[:500]}...
        **대화 문맥**: {conversation_context if conversation_context else "문맥 없음"}
        **지침**:
        - 로컬 문서를 먼저 찾는 데 꼼꼼하게 찾는 것이 좋음.
        - 각 항목은 한 문장 이상.
        - 대안을 무조건 알려줄 것.
        - 출처를 간단히 마지막에 명시.
        """
        try:
            print("Thought: Generating response with LLM...")
            response = self.llm.invoke(prompt)
            state["search_response"] = response.content
            print(f"Observation: Response generated: {state['search_response'][:100]}...")
        except Exception as e:
            print(f"Observation: LLM processing failed: {e}")
            state["search_response"] = f"검색 처리 중 오류: {e}"
        return state
    
class ConversationAgent:
    def __init__(self, memory_file="conversation_history.json"):
        self.llm = ChatOllama(model="gemma3:12b-it-q4_K_M")
        # self.llm = ChatOllama(model='qwen3:14b-q4_K_M')
        self.memory_file = memory_file
        self.conversation_history = self.load_conversation_history()
        # self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embed_model = global_embed_model
        self.update_faiss_index()
        self.tools = [
            Tool(
                name="ConversationHistorySearch",
                func=self.search_conversation_history,
                description="대화 기록에서 관련 정보를 검색합니다."
            )
        ]

    def load_conversation_history(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"[오류] 대화 기록 로드 실패: {e}")
            return []

    def save_conversation_history(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[오류] 대화 기록 저장 실패: {e}")

    def update_faiss_index(self):
        history_texts = [f"Q: {entry['user_query']}\\nA: {entry['agent_response']}" 
                         for entry in self.conversation_history]
        if history_texts:
            embeddings = self.embed_model.encode(history_texts)
            self.faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])
            self.faiss_index.add(np.array(embeddings))
            self.history_store = history_texts
        else:
            self.faiss_index = None
            self.history_store = []

    def search_conversation_history(self, query: str) -> str:
        print("Action: Searching conversation history...")
        if not self.history_store or not self.faiss_index:
            print("Observation: No conversation history available.")
            return "대화 기록이 없습니다."
        q_emb = self.embed_model.encode([query])
        D, I = self.faiss_index.search(np.array(q_emb), k=5)
        results = [self.history_store[i] for i in I[0] if i < len(self.history_store)]
        if not results:
            print("Observation: No relevant conversation history found.")
            return "관련 대화 기록을 찾지 못했습니다."
        print(f"Observation: Found {len(results)} relevant conversation entries.")
        return "\n\n".join(results)

    def update_conversation(self, state: AgentState) -> AgentState:
        query = state["query"]
        response = state["search_response"]
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": query,
            "agent_response": response,
            "source": "SearchAgent"
        }
        self.conversation_history.append(entry)
        self.save_conversation_history()
        self.update_faiss_index()
        state["conversation_context"] = self.get_conversation_context()
        return state

    def summarize_conversation(self, state: AgentState) -> AgentState:
        history_text = "\n".join([f"Q: {entry['user_query']}\nA: {entry['agent_response']}" 
                                  for entry in self.conversation_history[-5:]])  # 최근 5개로 제한
        prompt = f"""
        당신은 대화 기록을 요약하는 에이전트입니다. 다음 대화를 **친근하고 자연스러운 구어체**로 **한글로 존댓말을 사용해서** 간결히 요약해 주세요.  
        목록 형식으로 제공하세요.
        **대화 내용**:  
        {history_text if history_text else "대화 기록 없음"}
        **요약**:
        """
        try:
            print("Thought: Generating conversation summary...")
            history_result = self.search_conversation_history(state["query"])
            prompt += f"\n**추가 대화 기록 검색 결과**:\n{history_result[:300]}..."  # 검색 결과 제한
            summary = self.llm.invoke(prompt)
            state["conversation_summary"] = summary.content
            print(f"Observation: Summary generated: {state['conversation_summary'][:100]}...")
        except Exception as e:
            print(f"Observation: Conversation summary failed: {e}")
            state["conversation_summary"] = f"요약 생성 실패: {e}"
        return state

    def get_conversation_context(self):
        return "\n".join([f"Q: {entry['user_query']}\nA: {entry['agent_response']}" 
                          for entry in self.conversation_history[-3:]])
    
def router_node(state: AgentState) -> AgentState:
    return state

def router_condition(state: AgentState) -> str:
    summary_keywords = ["요약", "정리", "리뷰", "대화 내용", "지금까지", "이전 대화"]
    query = state["query"].lower()
    if any(keyword in query for keyword in summary_keywords):
        print("Thought: Detected summary intent in query.")
        return "conversation_agent_summarize"
    print("Thought: Defaulting to search intent.")
    return "search_agent"

def search_agent_node(state: AgentState) -> AgentState:
    search_agent = SearchAgent()  # 싱글톤으로 한 번만 초기화
    return search_agent.process_query(state)

def conversation_agent_update_node(state: AgentState) -> AgentState:
    conversation_agent = ConversationAgent()
    return conversation_agent.update_conversation(state)

def conversation_agent_summarize_node(state: AgentState) -> AgentState:
    conversation_agent = ConversationAgent()
    return conversation_agent.summarize_conversation(state)

def combine_node(state: AgentState) -> AgentState:
    final = []
    if state.get("conversation_summary"):
        final.append(f"대화 요약:\n{state['conversation_summary']}")
    if state.get("search_response"):
        final.append(f"검색 결과:\n{state['search_response']}")
    state["final_answer"] = "\n\n".join(final)
    return state

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("search_agent", search_agent_node)
graph.add_node("conversation_agent_update", conversation_agent_update_node)
graph.add_node("conversation_agent_summarize", conversation_agent_summarize_node)
graph.add_node("combine", combine_node)

graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    router_condition,
    {
        "search_agent": "search_agent",
        "conversation_agent_summarize": "conversation_agent_summarize"
    }
)
graph.add_edge("search_agent", "conversation_agent_update")
graph.add_edge("conversation_agent_update", "combine")
graph.add_edge("conversation_agent_summarize", "combine")
graph.set_finish_point("combine")

runnable_graph = graph.compile()

while True:
    user_input = input("\n🧑 사용자 질문: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("종료합니다.")
        break
    try:
        input_state = {
            "query": user_input,
            "conversation_context": None,
            "search_response": None,
            "conversation_summary": None,
            "final_answer": None
        }
        result = runnable_graph.invoke(input_state, {"debug": True})
        print(f"\n🤖 에이전트 응답:\n{result['final_answer']}")
    except Exception as e:
        print(f"[오류 발생] {e}")
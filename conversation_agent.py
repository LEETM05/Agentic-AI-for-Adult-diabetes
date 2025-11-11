import json, os, faiss, numpy as np
from datetime import datetime
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from langchain_core.tools import Tool
from state_definitions import AgentState

global_embed_model = SentenceTransformer('BAAI/bge-m3')

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
import os, json, fitz, faiss, numpy as np
from glob import glob
from langchain_ollama import ChatOllama
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool

global_embed_model = SentenceTransformer('BAAI/bge-m3')

class SearchAgent:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        self.llm = ChatOllama(model="gemma3:12b-it-q4_K_M")
        self.embed_model = global_embed_model
        self.search = DuckDuckGoSearchRun()
        print("PDF 로딩 및 청크 생성 중...")
        raw_docs = self.extract_texts_from_pdfs("./data")
        self.chunks = self.split_text_to_chunks(raw_docs)
        embeddings = self.embed_model.encode([d.page_content for d in self.chunks])
        self.faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])
        self.faiss_index.add(np.array(embeddings))
        self.chunk_store = self.chunks
        self.initialized = True
        self.tools = [
            Tool(name="LocalSearch", func=self.faiss_search, description="FAISS 기반 로컬 문서 검색"),
            Tool(name="WebSearch", func=self.web_search_tool_func, description="DuckDuckGo 기반 웹 검색")
        ]

    def extract_texts_from_pdfs(self, folder_path="./data"):
        pdf_paths = sorted(glob(os.path.join(folder_path, "*.pdf")))
        docs = []
        for path in pdf_paths:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    docs.append(Document(page_content=text.strip()))
        return docs

    def split_text_to_chunks(self, docs, chunk_size=1000, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[".", "\n"])
        return splitter.split_documents(docs)

    def faiss_search(self, query: str, threshold: float = 0.6, k: int = 7) -> str:
        q_emb = self.embed_model.encode([query])
        D, I = self.faiss_index.search(np.array(q_emb), k=k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.chunk_store): continue
            sim = 1 / (1 + dist)
            if sim < threshold: continue
            results.append(f"문서 {idx} (신뢰도 {sim:.2f}): {self.chunk_store[idx].page_content}")
        return "\n\n".join(results) if results else "로컬 DB에서 관련 문서를 찾지 못했습니다."

    def web_search_tool_func(self, query: str) -> str:
        try:
            return self.search.invoke(query)
        except Exception as e:
            return f"웹 검색 실패: {e}"

    def process_query(self, state):
        query = state["query"]
        local_result = self.faiss_search(query, threshold=0.57)
        web_result = (self.web_search_tool_func(query)
                      if "찾지 못했습니다" in local_result else "웹 검색 생략")
        prompt = f"""
        당신은 당뇨병 관리에 특화된 AI입니다.
        질문: {query}
        로컬 검색: {local_result[:500]}
        웹 검색: {web_result[:500]}
        """
        try:
            response = self.llm.invoke(prompt)
            state["search_response"] = response.content
        except Exception as e:
            state["search_response"] = f"오류: {e}"
        return state

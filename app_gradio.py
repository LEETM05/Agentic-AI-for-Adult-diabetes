# app_gradio.py
import sys, os
import gradio as gr
from graph_builder import build_graph

# 1. 모듈 경로 설정 (기존 main.py와 동일)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 그래프 빌드 (서버 시작 시 딱 한 번 실행됩니다)
#    - SearchAgent의 PDF 로딩, FAISS 인덱싱 등이 여기서 발생합니다.
print("🤖 AI 에이전트 그래프를 빌드하는 중... (모델 로딩 중)")
runnable_graph = build_graph()
print("✅ 그래프 빌드 완료. Gradio 서버를 시작합니다.")


# 3. Gradio가 호출할 함수 정의
def chat_with_agent(user_input, chat_history):
    """
    Gradio ChatInterface가 사용자 입력을 받을 때마다 이 함수를 호출합니다.
    chat_history는 Gradio가 관리하는 UI용 기록입니다.
    (참고: 우리가 만든 ConversationAgent는 파일(conversation_history.json)을
     통해 자체적으로 대화 기록을 관리하므로, 이 함수의 chat_history 파라미터를
     꼭 사용하지 않아도 됩니다.)
    """
    
    # 1. LangGraph에 전달할 상태(State) 객체 생성
    state = {
        "query": user_input,
        "conversation_context": None, # Agent가 자체적으로 관리/로드
        "search_response": None,
        "conversation_summary": None,
        "final_answer": None,
    }

    try:
        # 2. 그래프 실행
        print(f"🧑 사용자 질문: {user_input}")
        result = runnable_graph.invoke(state)
        
        # 3. 최종 답변 반환
        print(f"🤖 에이전트 응답: {result['final_answer']}")
        return result['final_answer']
    
    except Exception as e:
        print(f"[오류 발생] {e}")
        return f"죄송합니다, 처리 중 오류가 발생했습니다: {e}"

# 4. Gradio 채팅 인터페이스 실행
iface = gr.ChatInterface(
    fn=chat_with_agent,
    title="🧑‍⚕️ 당뇨병 관리 AI 에이전트",
    description="LangGraph와 FAISS, Ollama로 구축된 에이전트입니다.",
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="당뇨병에 대해 궁금한 점을 물어보세요...", container=False, scale=7),
    theme="soft",
    examples=[
        "당뇨병 초기 증상이 뭐야?",
        "오늘 대화 내용 요약해줘",
        "혈당 관리에 좋은 음식 알려줘"
    ],
    cache_examples=False # 상태가 있으므로 캐시 비활성화
)

# share=True로 설정하면 외부에서 접속 가능한 public URL이 생성됩니다.
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
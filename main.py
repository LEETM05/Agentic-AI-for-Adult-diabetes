# main.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 모듈 경로 보장

from graph_builder import build_graph

if __name__ == "__main__":
    runnable_graph = build_graph()
    while True:
        user_input = input("\n🧑 사용자 질문: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("종료합니다.")
            break
        state = {
            "query": user_input,
            "conversation_context": None,
            "search_response": None,
            "conversation_summary": None,
            "final_answer": None,
        }
        try:
            result = runnable_graph.invoke(state, {"debug": True})
            print(f"\n🤖 에이전트 응답:\n{result['final_answer']}")
        except Exception as e:
            print(f"[오류 발생] {e}")

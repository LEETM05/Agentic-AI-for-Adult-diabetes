# graph_builder.py
from langgraph.graph import StateGraph
from state_definitions import AgentState
from search_agent import SearchAgent
from conversation_agent import ConversationAgent

# --- node impls ---
def router_node(state: AgentState) -> AgentState:
    return state

def router_condition(state: AgentState) -> str:
    summary_keywords = ["요약", "정리", "리뷰", "대화 내용", "지금까지", "이전 대화"]
    q = (state.get("query") or "").lower()
    return "conversation_agent_summarize" if any(k in q for k in summary_keywords) else "search_agent"

def search_agent_node(state: AgentState) -> AgentState:
    return SearchAgent().process_query(state)

def conversation_agent_update_node(state: AgentState) -> AgentState:
    return ConversationAgent().update_conversation(state)

def conversation_agent_summarize_node(state: AgentState) -> AgentState:
    return ConversationAgent().summarize_conversation(state)

def combine_node(state: AgentState) -> AgentState:
    parts = []
    if state.get("conversation_summary"):
        parts.append(f"대화 요약:\n{state['conversation_summary']}")
    if state.get("search_response"):
        parts.append(f"검색 결과:\n{state['search_response']}")
    state["final_answer"] = "\n\n".join(parts) if parts else ""
    return state

# --- the function you need ---
def build_graph():
    graph = StateGraph(AgentState)

    # nodes
    graph.add_node("router", router_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("conversation_agent_update", conversation_agent_update_node)
    graph.add_node("conversation_agent_summarize", conversation_agent_summarize_node)
    graph.add_node("combine", combine_node)

    # entry
    graph.set_entry_point("router")

    # condition: route -> search or summarize
    graph.add_conditional_edges(
        "router",
        router_condition,
        {
            "search_agent": "search_agent",
            "conversation_agent_summarize": "conversation_agent_summarize",
        },
    )

    # linear flows to combine
    graph.add_edge("search_agent", "conversation_agent_update")
    graph.add_edge("conversation_agent_update", "combine")
    graph.add_edge("conversation_agent_summarize", "combine")

    # finish
    graph.set_finish_point("combine")

    return graph.compile()

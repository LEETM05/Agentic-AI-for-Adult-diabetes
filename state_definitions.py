from typing import TypedDict, Optional

class AgentState(TypedDict):
    query: str
    conversation_context: Optional[str]
    search_response: Optional[str]
    conversation_summary: Optional[str]
    final_answer: Optional[str]

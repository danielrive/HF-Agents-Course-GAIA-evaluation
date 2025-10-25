from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from enum import Enum

class AgentState(TypedDict):
    question: str  ## get question to analyze firsts before answer
    messages: Annotated[List[BaseMessage], add_messages]  # To keep message history 
    redefined_question: str  # Question after analysis, if modified
    question_type: str  # Type of question (e.g., definitional, statistical)
    selected_tools: List[str]  # Name of the selected tool, if any
    current_step: str  # Current step in the agent workflow
    final_answer: str
    input_file: Optional[str] # Contains file path to image
    reason: str  # Reason for the selected tool
    last_tool_results: str
    tool_loop_count: int


class AgentStep(Enum):
    ANALYZE_QUESTION = "analyze_question"
    AGENT_STEP = "agent_step"  # New: LLM reasoning with native tool calling (ReAct)
    COMPLETE = "complete"


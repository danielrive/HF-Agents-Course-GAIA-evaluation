from typing import Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from app.agent.state import AgentState, AgentStep
import logging
import re
import json
import uuid

# Tools
from app.tools import SearchTool
from app.tools.image_tool import analyze_image
from app.tools.audio_tool import transcribe_audio

logger = logging.getLogger(__name__)

# Define the LLM
llm = init_chat_model(
    "us.amazon.nova-premier-v1:0",
    model_provider="bedrock_converse"
)

## Analysis prompt
analysis_prompt = ChatPromptTemplate.from_template("""
You are analyzing a question to provide strategic guidance for answering it.

Question: {question}

Provide your analysis:

1. **Question Type:** Classify as: factual, statistical, comparative, causal, mathematical, or currency_convert

2. **Strategy:** What approach would work best for this question?
   - Can it be answered directly from general knowledge?
   - Does it need external verification (Wikipedia search)?
   - Does it require currency conversion?
   - Does it involve image or audio analysis?

3. **Refined Question:** Rephrase the question to be clear and specific.
   - Make it search-optimized if external lookup needed
   - Break into sub-questions if complex
   - Clean up if simple

4. **Reasoning:** Brief explanation of your strategy (1-2 sentences)

Focus on understanding what the question asks and how best to approach it.
""")


def init_agent_state(question: str) -> AgentState:
    logger.info(f"Initializing agent state for question: {question}")
    return AgentState(
        question=question,
        messages=[HumanMessage(content=question)],
        selected_tools=[],
        current_step=AgentStep.ANALYZE_QUESTION.value,
        final_answer="",
        input_file=None,
        question_type="",
        redefined_question="",
        reason="",
        tool_loop_count = 0,
        last_tool_results=[]
    )

# Tools
@tool
def web_search(question: str, question_type: str) -> str:
    """Search Wikipedia for information to answer questions."""
    return SearchTool().run(question, question_type)

@tool
def image_analyzer(image_path: str, question: str) -> str:
    """Analyze an image to extract information, read text, or answer questions about visual content."""
    return analyze_image(image_path, question)

@tool
def audio_transcriber(audio_path: str, language_code: str = "en-US") -> str:
    """Transcribe audio files to text. Supports MP3, WAV, MP4, FLAC, OGG, and other common formats."""
    return transcribe_audio(audio_path, language_code)

tools = [web_search, image_analyzer, audio_transcriber]
llm_with_tools = llm.bind_tools(tools)

# Pydantic Model for Structured Output
class AnalysisOutput(BaseModel):
    question_type: Literal["factual", "statistical", "comparative", "causal", "direct_answer", "mathematical"]
    strategy: str  # Strategic guidance for answering
    redefined_question: str
    reason: str

def analyze_question_node(state: AgentState) -> AgentState:
    """Analyze the question to provide strategic guidance."""
    try:
        logger.info(f"Analyzing question: {state['question']}")
        analysis_response = analysis_prompt | llm.with_structured_output(AnalysisOutput)
        results = analysis_response.invoke({"question": state["question"]})
        logger.info(f"Analysis complete - Type: {results.question_type}, Strategy: {results.strategy[:50]}...") 

        # Store analysis results for agent step
        state["question_type"] = results.question_type
        state["redefined_question"] = results.redefined_question
        state["reason"] = f"{results.strategy} | {results.reason}"
        
        logger.debug(f"Analysis: {results.model_dump()}")
    
    except Exception as e:
        logger.exception("Error during analysis")
        state["reason"] = f"Analysis failed: {str(e)}"
        state["redefined_question"] = state["question"]

    return state

## Agent step node with native tool calling (ReAct pattern)
def agent_step_node(state: AgentState) -> AgentState:
    """
    Main agent reasoning step with native LLM tool calling (ReAct pattern).
    LLM can call tools, see results, and iterate or provide final answer.
    """
    state["current_step"] = AgentStep.AGENT_STEP.value
    
    # System prompt with tool usage guidance
    system_prompt = """You are a precise assistant for the GAIA benchmark.

**Tool Usage Decision (Critical):**

Ask yourself: "Am I CERTAIN of the answer from my training?"

NO tools if:
- Common knowledge facts you're 100% certain about (e.g., capitals, basic math)
- Simple calculations you can do
- Standard definitions

USE tools if:
- Need to verify specific data (counts, dates, names, events)
- Question explicitly asks for Wikipedia or verified information
- Uncertain about the answer
- Question mentions specific people, albums, awards, etc. that need verification

**Available Tools:**
- web_search: Wikipedia search for factual information
- image_analyzer: Analyze images, OCR text from images
- audio_transcriber: Transcribe audio to text

**Output Format (CRITICAL for GAIA):**

When you have the answer, respond with ONLY the answer. Nothing else.

Examples:
Q: "How many albums?" → Answer: 3
Q: "What is the capital?" → Answer: Paris
Q: "Who won?" → Answer: John Smith


**Analysis Context:**
Question type: {question_type}
Strategy: {strategy}
Refined question: {refined_question}
{file_info}
"""
    
    # Build file info if available
    file_info = ""
    if state.get("input_file"):
        file_path = state["input_file"]
        if any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            file_info = f"\n**Available file:** Image at {file_path} (use image_analyzer tool)"
        elif any(file_path.endswith(ext) for ext in ['.mp3', '.wav', '.mp4', '.flac', '.ogg', '.m4a']):
            file_info = f"\n**Available file:** Audio at {file_path} (use audio_transcriber tool)"
    
    # Build context message with analysis guidance
    guidance_message = system_prompt.format(
        question_type=state.get('question_type', 'unknown'),
        strategy=state.get('reason', ''),
        refined_question=state.get('redefined_question', state['question']),
        file_info=file_info
    )
    
    # Add guidance as system-like message
    if state["tool_loop_count"] == 0:
        # First iteration - add guidance
        state["messages"].append(HumanMessage(content=guidance_message))
    
    # Let LLM decide: call tools or answer
    try:
        logger.info(f"Agent step #{state['tool_loop_count'] + 1}")
        
        # Safety check: max loop limit
        if state["tool_loop_count"] >= 5:
            logger.warning("Max tool loop limit reached (5), forcing completion")
            state["final_answer"] = "Max iterations reached"
            state["selected_tools"] = []
            return state
        
        response = llm_with_tools.invoke(state["messages"])
        state["messages"].append(response)
        
        # Check if LLM called tools or provided answer
        if response.tool_calls:
            logger.info(f"LLM called {len(response.tool_calls)} tools: {[tc['name'] for tc in response.tool_calls]}")
            state["selected_tools"] = [tc["name"] for tc in response.tool_calls]
            # Increment loop count when tools are called
            state["tool_loop_count"] += 1
        else:
            # LLM provided direct answer
            logger.info("LLM provided direct answer (no tool calls)")
            logger.debug(f"Response content type: {type(response.content)}, value: {response.content}")
            
            # Extract text from response
            content_text = response.content
            if isinstance(content_text, list):
                # If content is a list of dicts, extract text
                content_text = " ".join([item.get('text', '') for item in content_text if isinstance(item, dict)])
            elif not isinstance(content_text, str):
                content_text = str(content_text)
            
            cleaned_answer = clean_model_output(content_text)
            logger.debug(f"Cleaned answer: '{cleaned_answer}'")
            state["final_answer"] = cleaned_answer
            state["selected_tools"] = []
    
    except Exception as e:
        logger.exception("Error in agent step")
        error_msg = f"Error: {str(e)}"
        logger.error(f"Setting final_answer to: {error_msg}")
        state["final_answer"] = error_msg
        state["selected_tools"] = []
    
    return state

## Execute tools using LangChain's native ToolNode
def execute_tools_native_node(state: AgentState) -> AgentState:
    """Execute tools using LangChain's native ToolNode (no manual construction)."""
    state["current_step"] = "execute_tools"
    
    try:
        tool_node = ToolNode(tools)
        logger.info(f"Executing {len(state['selected_tools'])} tools via ToolNode")
        
        # ToolNode automatically handles tool execution and message formatting
        result = tool_node.invoke(state)
        
        # ToolNode returns updated messages with tool results
        state["messages"] = result["messages"]
        
        # Store tool results for reference
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        state["last_tool_results"] = [m.content[:200] + "..." if len(m.content) > 200 else m.content for m in tool_messages]
        
        logger.info(f"Executed {len(tool_messages)} tools successfully")
        logger.debug(f"Tool results: {state['last_tool_results']}")
        
    except Exception as e:
        logger.exception("Error executing tools")
        state["messages"].append(ToolMessage(
            content=f"Tool execution error: {str(e)}", 
            tool_call_id="error",
            name="error"
        ))
    
    return state

def clean_model_output(text: str) -> str:
    """
    Clean model output to extract just the answer for GAIA.
    Removes thinking tags, explanations, markdown, and extra formatting.
    """
    if not text:
        return text
    
    # Remove thinking tags
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    
    # Extract content from answer tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if answer_match:
        text = answer_match.group(1)
    
    # Strip whitespace early to avoid empty string issues
    text = text.strip()
    
    # Remove common prefixes
    text = re.sub(r"^(Answer:|Final Answer:|The answer is:?|Based on.*?,)\s*", "", text, flags=re.IGNORECASE)
    
    # Remove markdown bold/italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    
    # If there are bullet points or lists, just take the first line
    if re.search(r"[-•*]\s", text) or "\n" in text:
        # Take only the first line before any list/explanation
        first_line = text.split("\n")[0].strip()
        # If first line has bullets, extract the content
        first_line = re.sub(r"^[-•*]\s*", "", first_line)
        text = first_line
    
    # For verbose answers with numbers, try to extract just the number
    # Pattern: "Mercedes Sosa published 3 studio albums" → extract "3"
    if len(text) > 15:  # If answer is verbose
        # Check if it's a numeric answer with extra words
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if number_match:
            # Check if the text clearly indicates this is THE answer
            if any(keyword in text.lower() for keyword in ['published', 'released', 'has', 'were', 'are', 'total', 'count']):
                # Just extract the first number as the answer
                text = number_match.group(1)
    
    return text.strip()

def complete_node(state: AgentState) -> AgentState:
    """Format final answer (AgentStep.COMPLETE)."""
    state["current_step"] = AgentStep.COMPLETE.value
    
    # Don't add a message with "Final answer:" prefix - it can pollute the actual answer
    # Just log it for debugging
    logger.info(f"Agent complete. Final answer: {state['final_answer']}")
    return state

### --- Build the Graph ---

graph_builder = StateGraph(AgentState)

# Nodes
graph_builder.add_node("analyze_question", analyze_question_node)
graph_builder.add_node("agent_step", agent_step_node)
graph_builder.add_node("execute_tools", execute_tools_native_node)
graph_builder.add_node("complete", complete_node)

# Flow
graph_builder.add_edge(START, "analyze_question")
graph_builder.add_edge("analyze_question", "agent_step")

# Conditional: did LLM call tools?
def route_from_agent_step(state: AgentState):
    if state.get("selected_tools"):
        return "execute_tools"
    return "complete"

graph_builder.add_conditional_edges(
    "agent_step",
    route_from_agent_step,
    {"execute_tools": "execute_tools", "complete": "complete"}
)

# After tools, loop back to agent_step (ReAct pattern)
graph_builder.add_edge("execute_tools", "agent_step")

graph_builder.add_edge("complete", END)

# Compile
graph = graph_builder.compile()

def run_agent(question: str) -> AgentState:
    state = init_agent_state(question)
    result = graph.invoke(state)
    
    # Clean output - only log final answer
    logger.info(f"Final Answer: {result['final_answer']}")
    return result

def draw_graph():
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))

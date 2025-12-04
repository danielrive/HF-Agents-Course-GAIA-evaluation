"""
Simple testing interface for the GAIA agent.
Allows manual testing of individual questions with debugging information.
"""

import logging
import time
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from app.agent.agent import run_agent, init_agent_state, graph


def format_debug_info(state: dict) -> str:
    """Format agent state into readable debug information."""
    debug_lines = []
    
    # Question info
    debug_lines.append("## Question Analysis")
    debug_lines.append(f"- **Question Type:** {state.get('question_type', 'N/A')}")
    debug_lines.append(f"- **Refined Question:** {state.get('redefined_question', 'N/A')}")
    debug_lines.append(f"- **Strategy:** {state.get('reason', 'N/A')}")
    debug_lines.append("")
    
    # Tools used
    tools_used = state.get('selected_tools', [])
    if tools_used:
        debug_lines.append("## Tools Used")
        for tool in tools_used:
            debug_lines.append(f"- {tool}")
        debug_lines.append("")
    
    # Tool results
    tool_results = state.get('last_tool_results', [])
    if tool_results:
        debug_lines.append("## Tool Results")
        # Handle both list and string formats
        if isinstance(tool_results, list):
            for i, result in enumerate(tool_results, 1):
                result_str = str(result)
                debug_lines.append(f"**Result {i}:** {result_str[:200]}...")
        else:
            debug_lines.append(f"**Results:** {str(tool_results)[:200]}...")
        debug_lines.append("")
    
    # Conversation steps
    messages = state.get('messages', [])
    if messages:
        debug_lines.append("## Conversation Steps")
        for i, msg in enumerate(messages[-5:], 1):  # Show last 5 messages
            msg_type = type(msg).__name__
            content = str(msg.content)[:150]
            debug_lines.append(f"**Step {i}** ({msg_type}): {content}...")
        debug_lines.append("")
    
    # Current step
    debug_lines.append(f"## Status")
    debug_lines.append(f"- **Current Step:** {state.get('current_step', 'N/A')}")
    debug_lines.append(f"- **Tool Loop Count:** {state.get('tool_loop_count', 0)}")
    
    # File info
    if state.get('input_file'):
        debug_lines.append(f"- **Input File:** {state.get('input_file')}")
    
    return "\n".join(debug_lines)


def test_agent(
    question: str,
    uploaded_file: Optional[gr.File] = None
) -> Tuple[str, str]:
    """
    Test the agent with a question and optional file.
    
    Args:
        question: The question to ask
        uploaded_file: Optional file (image or audio)
        
    Returns:
        Tuple of (final_answer, debug_info)
    """
    if not question or not question.strip():
        return "Please enter a question.", ""
    
    start_time = time.time()
    
    try:
        logger.info(f"Testing agent with question: {question[:100]}...")
        
        # Initialize state
        state = init_agent_state(question.strip())
        
        # Handle file upload if provided
        if uploaded_file is not None:
            # Gradio file upload returns a file path string
            if isinstance(uploaded_file, str):
                file_path = uploaded_file
            elif hasattr(uploaded_file, 'name'):
                file_path = uploaded_file.name
            else:
                file_path = str(uploaded_file)
            
            # Verify file exists
            if not Path(file_path).exists():
                return f"Error: File not found at {file_path}", ""
            
            logger.info(f"Processing with file: {file_path}")
            state["input_file"] = file_path
        
        # Run the agent - same as GAIA runner
        result = graph.invoke(state)
        
        # Extract answer - same logic as app/gaia/runner.py
        answer = result.get("final_answer", "")
        if not answer or answer.startswith("Error"):
            # Fallback to last message
            if result.get("messages"):
                answer = str(result["messages"][-1].content)
            else:
                answer = "No answer generated."
        
        # Format debug info
        debug_info = format_debug_info(result)
        
        # Add execution time
        execution_time = time.time() - start_time
        debug_info = f"**Execution Time:** {execution_time:.2f} seconds\n\n" + debug_info
        
        logger.info(f"Agent completed in {execution_time:.2f}s. Answer: {answer[:100]}...")
        
        return answer, debug_info
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.exception(f"Exception in test_agent: {e}")
        debug_info = f"**Error occurred:**\n```\n{error_msg}\n```"
        return error_msg, debug_info


# Create Gradio interface
with gr.Blocks(title="Agent Testing Interface") as demo:
    gr.Markdown("""
    # 🤖 Agent Testing Interface
    
    Test your GAIA agent with individual questions and see detailed debugging information.
    
    **Features:**
    - Test single questions
    - Upload images or audio files
    - View step-by-step execution
    - See which tools are used
    - Debug errors easily
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Question",
                placeholder="Enter your question here...",
                lines=3,
                interactive=True
            )
            
            file_input = gr.File(
                label="Optional File (Image or Audio)",
                file_types=["image", "audio"],
                type="filepath"
            )
            
            run_button = gr.Button("Run Agent", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Final Answer",
                lines=5,
                interactive=False
            )
    
    with gr.Accordion("🔍 Debug Information", open=False):
        debug_output = gr.Markdown(
            label="Debug Info",
            value="Run the agent to see debug information here."
        )
    
    # Examples
    gr.Markdown("### Example Questions")
    examples = gr.Examples(
        examples=[
            ["What is the capital of France?"],
            ["How many albums did Mercedes Sosa publish?"],
            ["What is 2 + 2?"],
        ],
        inputs=question_input
    )
    
    # Connect the button
    run_button.click(
        fn=test_agent,
        inputs=[question_input, file_input],
        outputs=[answer_output, debug_output]
    )
    
    # Also run on Enter key
    question_input.submit(
        fn=test_agent,
        inputs=[question_input, file_input],
        outputs=[answer_output, debug_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main.py
        share=False,
        debug=True
    )


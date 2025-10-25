import os
import logging
import gradio as gr
import requests
import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from app.agent.agent import run_agent, init_agent_state, graph 

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs, INFO for normal
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def clean_answer(text: str) -> str:
    """
    Remove <thinking> or similar tags from the model output.
    Keep only the final user-facing answer.
    """
    # remove <thinking>...</thinking> sections
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # strip whitespace
    return text.strip()

class GAIAAgent:
    def __init__(self):
        logger.info("GAIAAgent initialized.")
        self.downloads_dir = Path("./gaia_downloads")
        self.downloads_dir.mkdir(exist_ok=True)

    def download_file(self, url: str, task_id: str) -> str: 
        """Download a file from URL and return local path."""
        try:
            parsed = urlparse(url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"{task_id}_file"

            local_path = self.downloads_dir / f"{task_id}_{filename}"
            
            if not local_path.exists():
                logger.info(f"Downloading file: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded to: {local_path}")
            else:
                logger.info(f"Using cached file: {local_path}")
            
            return str(local_path)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None

    def __call__(self, question: str, file_url: str = None, file_path: str = None, task_id: str = None) -> str:
        """
        GAIA will call this function with a question and optionally a file.
        We pass it to our LangGraph agent and return the answer.
        
        Args:
            question: The question text
            file_url: URL to download file from (if provided by GAIA)
            file_path: Local file path (if provided by GAIA)
            task_id: Task ID for file naming
        """
        logger.info(f"GAIAAgent received question: {question[:50]}...")
        
        try:
            input_file = None
            
            if file_url:
                logger.info(f"File URL provided: {file_url}")
                input_file = self.download_file(file_url, task_id or "unknown")
            elif file_path:
                logger.info(f"File path provided: {file_path}")
                input_file = file_path
            
            # Process question with or without file
            if input_file:
                logger.info(f"Processing with input_file: {input_file}")
                state = init_agent_state(question)
                state["input_file"] = input_file
                response = graph.invoke(state)
            else:
                logger.info("Processing text-only question")
                response = run_agent(question)
            # Extract and clean answer
            answer = response.get("final_answer", "")
            logger.debug(f"Raw final_answer: '{answer}'")
            logger.debug(f"Response keys: {list(response.keys())}")
            if not answer or answer.startswith("Error"):
                raw_answer = response["messages"][-1].content
                logger.info(f"Fallback to last message: '{raw_answer}'")
                logger.debug(f"All messages: {[msg.content for msg in response['messages']]}")
                answer = clean_answer(raw_answer)
            else:
                answer = clean_answer(answer)
            logger.info(f"Cleaned answer: '{answer}'")
            
        except Exception as e:
            answer = f"Error: {e}"
            logger.exception(f"Exception: {e}")
            
        logger.info(f"Returning: {answer[:100]}...")
        return answer


def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    
    if not space_id:
        return "Error: SPACE_ID environment variable not set", None

    if profile:
        username = f"{profile.username}"
        logger.info(f"User logged in: {username}")
    else:
        return "Please Login to Hugging Face.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GAIAAgent()
    except Exception as e:
        logger.exception(f"Error initializing agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    logger.info(f"Agent code: {agent_code}")

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        logger.exception(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    results_log, answers_payload = [], []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        
        # Check for files - GAIA provides file_name field
        file_name = item.get("file_name", "").strip()
        
        # Construct file URL from GAIA server if file_name exists
        file_url = None
        if file_name:
            # GAIA serves files at: {api_url}/files/{filename}
            file_url = f"{api_url}/files/{file_name}"
            logger.info(f"Task {task_id} - File detected: {file_name}, URL: {file_url}")
        
        file_path = item.get("file_path") or item.get("local_file")
        
        logger.debug(f"Task {task_id} - file_url: {file_url}, file_path: {file_path}")
        
        # Call agent with file information
        submitted_answer = agent(
            question=question_text,
            file_url=file_url,
            file_path=file_path,
            task_id=task_id
        )
        
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        
        # Log with file info
        log_entry = {
            "Task ID": task_id,
            "Question": question_text[:100] + "..." if len(question_text) > 100 else question_text,
            "Has File": "Yes" if (file_url or file_path) else "No",
            "Submitted Answer": submitted_answer[:100] + "..." if len(submitted_answer) > 100 else submitted_answer
        }
        results_log.append(log_entry)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"✅ Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
        )
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)


# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation")
    
    # Simple login button - let Gradio handle OAuth
    login_button = gr.LoginButton()
    
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )


if __name__ == "__main__":
    demo.launch(debug=False, share=False)

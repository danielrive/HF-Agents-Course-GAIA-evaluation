import base64
import logging
from pathlib import Path
from typing import Optional
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


def analyze_image(image_path: str, question: str) -> str:
    """
    Analyze an image using AWS Bedrock Nova.
    
    Supports common image formats: JPEG, PNG, GIF, WebP
    
    Args:
        image_path: Path to the image file (relative or absolute)
        question: Question to ask about the image or instructions for analysis
        
    Returns:
        Analysis results as text
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is not supported
    """
    logger.info(f"Analyzing image: {image_path}")
    logger.debug(f"Question: {question}")
    
    # Validate file exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine image format from file extension
    suffix = image_file.suffix.lower()
    format_map = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".gif": "gif",
        ".webp": "webp"
    }
    
    image_format = format_map.get(suffix)
    if not image_format:
        raise ValueError(f"Unsupported image format: {suffix}. Supported: {list(format_map.keys())}")
    
    logger.debug(f"Image format: {image_format}")
    
    # Load and encode image to base64
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        logger.info(f"Loaded image: {len(image_bytes)} bytes")
        
        # Create multimodal message for Bedrock Converse API
        # Format: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                },
                {
                    "text": question
                }
            ]
        }
        
        # Initialize Bedrock Nova Premier (supports vision)
        llm = init_chat_model(
            "us.amazon.nova-premier-v1:0",
            model_provider="bedrock_converse"
        )
        
        logger.debug("Sending image to Bedrock Nova Premier...")
        response = llm.invoke([message])
        
        result = response.content
        logger.info(f"Image analysis complete. Response length: {len(result)} chars")
        logger.debug(f"Response: {result[:200]}...")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error analyzing image: {e}")
        raise


def extract_text_from_image(image_path: str) -> str:
    """
    Extract all text from an image (OCR).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text
    """
    question = """Extract all text from this image. 
    Return only the text content, preserving the layout and formatting as much as possible.
    If there is no text, say "No text found"."""
    
    return analyze_image(image_path, question)


def describe_image(image_path: str) -> str:
    """
    Get a detailed description of what's in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Detailed description
    """
    question = """Describe this image in detail. 
    Include what you see, any text, numbers, objects, people, and anything notable."""
    
    return analyze_image(image_path, question)


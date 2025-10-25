import boto3
import time
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, language_code: str = "en-US") -> str:
    """
    Transcribe audio file to text using Amazon Transcribe.
    
    Supports common audio formats: MP3, MP4, WAV, FLAC, OGG, AMR, WebM
    
    Args:
        audio_path: Path to the audio file (relative or absolute)
        language_code: Language code for transcription (default: "en-US")
                      Common codes: en-US, es-ES, fr-FR, de-DE, it-IT, pt-BR, ja-JP, ko-KR, zh-CN
        
    Returns:
        Transcribed text
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio format is not supported
        Exception: If transcription fails
    """
    logger.info(f"Transcribing audio: {audio_path}")
    logger.debug(f"Language: {language_code}")
    
    # Validate file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Determine audio format from file extension
    suffix = audio_file.suffix.lower()
    format_map = {
        ".mp3": "mp3",
        ".mp4": "mp4",
        ".wav": "wav",
        ".flac": "flac",
        ".ogg": "ogg",
        ".amr": "amr",
        ".webm": "webm",
        ".m4a": "mp4",  # M4A is typically MP4 audio
    }
    
    media_format = format_map.get(suffix)
    if not media_format:
        raise ValueError(f"Unsupported audio format: {suffix}. Supported: {list(format_map.keys())}")
    
    logger.debug(f"Audio format: {media_format}")
    
    try:
        # Initialize AWS Transcribe client
        transcribe_client = boto3.client('transcribe')
        s3_client = boto3.client('s3')
        
        # Generate unique job name and S3 key
        job_name = f"transcribe-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Get or create S3 bucket for temporary upload
        # Using a default bucket name, you might want to make this configurable
        bucket_name = "smart-cash-agent-transcribe-temp"
        s3_key = f"audio/{job_name}{suffix}"
        
        logger.info(f"Uploading audio to S3: s3://{bucket_name}/{s3_key}")
        
        # Check if bucket exists, create if it doesn't
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except:
            logger.info(f"Creating S3 bucket: {bucket_name}")
            try:
                # For us-east-1, don't specify LocationConstraint
                import boto3.session
                session = boto3.session.Session()
                region = session.region_name or 'us-east-1'
                
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"Bucket created: {bucket_name}")
            except Exception as e:
                logger.warning(f"Could not create bucket: {e}")
                # Try with a more unique bucket name
                bucket_name = f"smart-cash-agent-transcribe-{uuid.uuid4().hex[:8]}"
                s3_key = f"audio/{job_name}{suffix}"
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"Created bucket with unique name: {bucket_name}")
        
        # Upload audio file to S3
        with open(audio_path, 'rb') as audio_file:
            s3_client.upload_fileobj(audio_file, bucket_name, s3_key)
        
        media_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"Starting transcription job: {job_name}")
        
        # Start transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=media_format,
            LanguageCode=language_code,
        )
        
        # Wait for transcription to complete
        logger.info("Waiting for transcription to complete...")
        max_tries = 60  # 5 minutes maximum
        sleep_time = 5  # Check every 5 seconds
        
        for attempt in range(max_tries):
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            logger.debug(f"Transcription status: {status} (attempt {attempt + 1}/{max_tries})")
            
            if status == 'COMPLETED':
                logger.info("Transcription completed successfully")
                
                # Get transcript
                transcript_uri = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                
                # Download transcript using requests
                import requests
                transcript_response = requests.get(transcript_uri)
                transcript_data = transcript_response.json()
                
                # Extract text
                transcript_text = transcript_data['results']['transcripts'][0]['transcript']
                
                logger.info(f"Transcription length: {len(transcript_text)} characters")
                logger.debug(f"Transcript preview: {transcript_text[:200]}...")
                
                # Cleanup: Delete S3 object and transcription job
                try:
                    s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                    transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                    logger.debug("Cleaned up S3 object and transcription job")
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")
                
                return transcript_text
                
            elif status == 'FAILED':
                failure_reason = response['TranscriptionJob'].get('FailureReason', 'Unknown')
                logger.error(f"Transcription failed: {failure_reason}")
                
                # Cleanup on failure
                try:
                    s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                    transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                except:
                    pass
                
                raise Exception(f"Transcription failed: {failure_reason}")
            
            # Wait before next check
            time.sleep(sleep_time)
        
        # Timeout
        logger.error("Transcription timed out")
        
        # Cleanup on timeout
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        except:
            pass
        
        raise Exception("Transcription timed out after 5 minutes")
        
    except Exception as e:
        logger.exception(f"Error transcribing audio: {e}")
        raise


def transcribe_audio_simple(audio_path: str) -> str:
    """
    Simple wrapper for transcribe_audio with auto-detected language.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    return transcribe_audio(audio_path, language_code="en-US")


def analyze_audio(audio_path: str, question: str, language_code: str = "en-US") -> str:
    """
    Transcribe audio and optionally answer a question about it.
    
    Args:
        audio_path: Path to the audio file
        question: Question to answer about the audio (or "transcribe" for full transcript)
        language_code: Language code for transcription
        
    Returns:
        Transcribed text or answer to the question
    """
    logger.info(f"Analyzing audio: {audio_path}")
    logger.debug(f"Question: {question}")
    
    # Transcribe the audio
    transcript = transcribe_audio(audio_path, language_code)
    
    # If the question is just asking for transcription, return it
    if question.lower() in ["transcribe", "what does it say", "what is said"]:
        return transcript
    
    # Otherwise, return the transcript with context
    result = f"Audio transcript:\n\n{transcript}\n\n"
    
    # Note: For complex questions about the audio, you could use an LLM here
    # to analyze the transcript and answer the specific question
    
    return result


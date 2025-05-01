from fastapi import APIRouter, HTTPException
import os
from fastapi import File, UploadFile, HTTPException
from pathlib import Path
import logging
from datetime import datetime, timedelta
from .text_extraction import extract_and_clean_text
from .main import ResumeParser
from Expericecal.total_exp import format_experience, calculator
from database.operations import ResumeOperations
from database.client import get_collection
from core.vectorizer import Vectorizer

# Initialize your parser with API keys (replace with your actual keys)
API_KEYS = [
    "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",
    # Add more keys if you have
]

parser = ResumeParser(API_KEYS)
collection = get_collection()
vectorizer = Vectorizer()
resume_ops = ResumeOperations(collection, vectorizer)
# Create a router instance
router = APIRouter()


BASE_FOLDER = "dummy_data_save"
TEMP_FOLDER = os.path.join(BASE_FOLDER, "temp_text_extract")
TEMP_DIR = Path(os.path.join(BASE_FOLDER, "temp_files"))

# Ensure the directories exist
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="cleanup.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cleanup_temp_directory(age_limit_minutes: int = 60):
    """
    Cleanup temporary directory by deleting files older than the specified age limit.
    """
    now = datetime.now()
    for file_path in TEMP_DIR.iterdir():
        if file_path.is_file():
            file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age > timedelta(minutes=age_limit_minutes):
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")


@router.post("/grouqcloud/")
async def extract_clean_text_llam3_3b(file: UploadFile = File(...)):
    """
    Endpoint to extract and clean text from uploaded file for llm model.
    """
    try:
        # Define the path to save the uploaded file
        file_location = os.path.join(TEMP_FOLDER, file.filename)

        # Save the uploaded file
        with open(file_location, "wb") as temp_file:
            temp_file.write(await file.read())

        # Extract text
        _, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in [".txt", ".pdf", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only .txt, .pdf, and .docx are supported.",
            )

        cleaned_text = extract_and_clean_text(file_location)
        print(cleaned_text)
        resume_parser = parser.process_resume(cleaned_text)
        print(resume_parser)
        if "total_experience" not in resume_parser:
            resume_parser["total_experience"] = 0
        print(resume_parser["total_experience"])
        resume_parser_exp = resume_parser["experience"]
        res = calculator.calculate_experience(resume_parser)
        resume_parser["total_experience"] = format_experience(res[0], res[1])

        resume_ops.create_resume(resume_parser)

        # Delete the temporary file
        os.remove(file_location)

        return {
            "filename": file.filename,
            "cleaned_text": cleaned_text,
            "resume_parser": resume_parser,
            # "resume_parser_exp": resume_parser_exp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

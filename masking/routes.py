from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import os
import uuid
import logging
from datetime import datetime, timedelta
from .maskingut import mask_pdf, mask_docx, mask_txt

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


@router.post("/mask-resume/", tags=["resume_masker"])
async def mask_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to mask sensitive data in uploaded resumes.
    """
    input_filename = file.filename
    ext = os.path.splitext(input_filename)[1].lower()

    unique_id = uuid.uuid4()
    temp_input = TEMP_DIR / f"{unique_id}_{input_filename}"
    output_filename = f"masked_{input_filename}"
    temp_output = TEMP_DIR / output_filename

    # Save uploaded file
    with open(temp_input, "wb") as f:
        f.write(await file.read())

    try:
        # Determine file type and mask accordingly
        if ext == ".pdf":
            mask_pdf(temp_input, temp_output)
        elif ext == ".docx":
            mask_docx(temp_input, temp_output)
        elif ext == ".txt":
            mask_txt(temp_input, temp_output)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Schedule cleanup of files
        background_tasks.add_task(temp_input.unlink)
        background_tasks.add_task(temp_output.unlink)

        # Perform directory cleanup
        background_tasks.add_task(cleanup_temp_directory, 60)

        # Return the masked file
        return FileResponse(
            path=temp_output,
            filename=output_filename,
            media_type="application/octet-stream",
        )
    except Exception as e:
        # Cleanup input file if masking fails
        if temp_input.exists():
            temp_input.unlink()
        raise HTTPException(status_code=500, detail=str(e))

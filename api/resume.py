# resume_api/api/resume.py
from fastapi import APIRouter, Depends, Body
from models.resume import ResumeCreate, ResumeUpdate
from database.operations import ResumeOperations
from core.vectorizer import Vectorizer
from database.client import get_collection
from typing import List, Dict

router = APIRouter(prefix="/resumes", tags=["Resumes"])

# Initialize dependencies
collection = get_collection()
vectorizer = Vectorizer()
resume_ops = ResumeOperations(collection, vectorizer)


@router.post("/", response_model=Dict[str, str])
async def create_resume(resume: ResumeCreate):
    return resume_ops.create_resume(resume.dict())


@router.put("/{resume_id}", response_model=Dict[str, str])
async def update_resume(resume_id: str, resume: ResumeUpdate):
    return resume_ops.update_resume(resume_id, resume.dict())


@router.get("/{resume_id}", response_model=Dict)
async def get_resume(resume_id: str):
    return resume_ops.get_resume(resume_id)


@router.delete("/{resume_id}", response_model=Dict)
async def delete_resume(resume_id: str):
    return resume_ops.delete_resume(resume_id)


@router.get("/", response_model=List[Dict])
async def list_resumes(skip: int = 0, limit: int = 10):
    return resume_ops.list_resumes(skip, limit)


@router.post("/update-all-vector-embeddings")
async def update_all_vector_embeddings():
    return resume_ops.update_all_vector_embeddings()

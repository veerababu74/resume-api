# resume_api/database/operations.py
from pymongo.collection import Collection
from bson import ObjectId
from typing import Dict, Any
from core.vectorizer import Vectorizer
from core.helpers import format_resume
from fastapi import HTTPException
from typing import List


class ResumeOperations:
    def __init__(self, collection: Collection, vectorizer: Vectorizer):
        self.collection = collection
        self.vectorizer = vectorizer

    def create_resume(self, resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a new resume with vector embeddings"""
        resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume_data)
        result = self.collection.insert_one(resume_with_vectors)
        return {
            "id": str(result.inserted_id),
            "message": "Resume created successfully with vector embeddings",
        }

    def update_resume(
        self, resume_id: str, resume_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Update a resume by ID with vector embeddings"""
        existing = self.collection.find_one({"_id": ObjectId(resume_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Resume not found")

        resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume_data)
        result = self.collection.update_one(
            {"_id": ObjectId(resume_id)}, {"$set": resume_with_vectors}
        )

        if result.modified_count == 1:
            return {"message": "Resume updated successfully with vector embeddings"}
        return {"message": "No changes made to the resume"}

    def get_resume(self, resume_id: str) -> Dict:
        """Get a resume by ID"""
        resume = self.collection.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        return format_resume(resume)

    def delete_resume(self, resume_id: str) -> Dict:
        """Delete a resume by ID"""
        result = self.collection.delete_one({"_id": ObjectId(resume_id)})
        if result.deleted_count != 1:
            raise HTTPException(status_code=404, detail="Resume not found")
        return {"message": "Resume deleted successfully"}

    def list_resumes(self, skip: int = 0, limit: int = 10) -> List[Dict]:
        """List all resumes with pagination"""
        cursor = self.collection.find().skip(skip).limit(limit)
        return [format_resume(doc) for doc in cursor]

    def update_all_vector_embeddings(self) -> Dict:
        """Update vector embeddings for all resumes"""
        resumes = list(self.collection.find({}))
        updated_count = 0

        for resume in resumes:
            resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume)
            result = self.collection.update_one(
                {"_id": resume["_id"]},
                {
                    "$set": {
                        "skills_vector": resume_with_vectors.get("skills_vector"),
                        "experience_text_vector": resume_with_vectors.get(
                            "experience_text_vector"
                        ),
                        "education_text_vector": resume_with_vectors.get(
                            "education_text_vector"
                        ),
                        "projects_text_vector": resume_with_vectors.get(
                            "projects_text_vector"
                        ),
                        "total_resume_vector": resume_with_vectors.get(
                            "total_resume_vector"
                        ),
                    }
                },
            )
            if result.modified_count > 0:
                updated_count += 1

        return {"message": f"Updated vector embeddings for {updated_count} resumes"}

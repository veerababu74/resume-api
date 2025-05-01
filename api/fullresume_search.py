from fastapi import FastAPI, HTTPException, Depends, Query, APIRouter, Body
from typing import List, Dict, Any
from core.vectorizer import Vectorizer
from database.client import get_collection
from core.helpers import format_resume

collection = get_collection()
vectorizer = Vectorizer()
router = APIRouter(prefix="/vector-search", tags=["Full Resume Search"])


@router.post("/full-resume/", response_model=List[Dict[str, Any]])
async def full_resume_search(
    query: str = Body(..., description="Search query for the entire resume"),
    num_results: int = Body(5, description="Number of results to return"),
):
    """Perform semantic search on the entire content of resumes"""
    try:
        # Generate embedding for search query
        query_embedding = vectorizer.generate_embedding(query)

        # Perform vector search on total_resume_vector
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "total_resume_vector",
                        "k": num_results,
                    },
                }
            },
            {
                "$project": {
                    "name": 1,
                    "contact_details": 1,
                    "education": 1,
                    "experience": 1,
                    "projects": 1,
                    "total_experience": 1,
                    "skills": 1,
                    "certifications": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        results = list(collection.aggregate(pipeline))
        formatted_results = [format_resume(result) for result in results]

        return formatted_results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Full resume search failed: {str(e)}"
        )


# 4. Add endpoint to update all resumes with the new total resume vector
@router.post("/update-total-resume-vectors/")
async def update_total_resume_vectors():
    """Update total_resume_vector for all resumes in the database"""
    try:
        # Get all resumes
        resumes = list(collection.find({}))

        updated_count = 0
        for resume in resumes:
            # Generate the total resume text
            name = resume.get("name", "N/A")
            total_experience = resume.get("total_experience", "")

            # Skills text
            skills = resume.get("skills", [])
            skills_text = ", ".join(skills) if isinstance(skills, list) else skills

            # Experience text
            experience_texts = []
            for exp in resume.get("experience", []):
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('duration', '')}"
                experience_texts.append(exp_text)
            experience_text = ". ".join(experience_texts)

            # Education text
            education_texts = []
            for edu in resume.get("education", []):
                edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('dates', '')})"
                education_texts.append(edu_text)
            education_text = ". ".join(education_texts)

            # Projects text
            project_texts = []
            for proj in resume.get("projects", []):
                tech_text = (
                    ", ".join(proj.get("technologies", []))
                    if isinstance(proj.get("technologies"), list)
                    else proj.get("technologies", "")
                )
                proj_text = f"{proj.get('name', '')}: {proj.get('description', '')}. Technologies: {tech_text}. Role: {proj.get('role', '')}"
                project_texts.append(proj_text)
            projects_text = ". ".join(project_texts)

            # Certifications text
            certifications = resume.get("certifications", [])
            certifications_text = (
                ", ".join(certifications)
                if isinstance(certifications, list)
                else certifications
            )

            # Combined text
            total_resume_text = f"Name: {name}. "
            if total_experience:
                total_resume_text += f"Total Experience: {total_experience}. "
            if skills_text:
                total_resume_text += f"Skills: {skills_text}. "
            if experience_text:
                total_resume_text += f"Experience: {experience_text}. "
            if education_text:
                total_resume_text += f"Education: {education_text}. "
            if projects_text:
                total_resume_text += f"Projects: {projects_text}. "
            if certifications_text:
                total_resume_text += f"Certifications: {certifications_text}."

            # Generate vector for the entire resume
            total_resume_vector = vectorizer.generate_embedding(total_resume_text)

            # Update in database
            result = collection.update_one(
                {"_id": resume["_id"]},
                {"$set": {"total_resume_vector": total_resume_vector}},
            )

            if result.modified_count > 0:
                updated_count += 1

        return {"message": f"Updated total resume vectors for {updated_count} resumes"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update total resume vectors: {str(e)}"
        )


# 5. Add a weighted multi-vector search option
@router.post("/vector-search/weighted/", response_model=List[Dict[str, Any]])
async def weighted_vector_search(
    query: str = Body(..., description="Search query"),
    weights: Dict[str, float] = Body(
        {
            "skills": 1.0,
            "experience": 1.0,
            "education": 1.0,
            "projects": 1.0,
            "full_resume": 2.0,  # By default, give more weight to the full resume match
        },
        description="Weights for different vector fields (values between 0.0 and 5.0)",
    ),
    num_results: int = Body(5, description="Number of results to return"),
):
    """
    Perform weighted semantic search across multiple vector fields
    """
    try:
        # Generate embedding for search query
        query_embedding = vectorizer.generate_embedding(query)

        # Validate weights
        valid_fields = {"skills", "experience", "education", "projects", "full_resume"}
        for field in weights:
            if field not in valid_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid field: {field}. Valid fields are: {', '.join(valid_fields)}",
                )

            if not 0.0 <= weights[field] <= 5.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Weight for {field} must be between 0.0 and 5.0",
                )

        # Map field names to vector fields
        field_to_vector = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",
            "projects": "projects_text_vector",
            "full_resume": "total_resume_vector",
        }

        # Build search stages for each weighted field
        search_results = []

        for field, weight in weights.items():
            if weight > 0:
                vector_field = field_to_vector[field]

                # Perform vector search
                pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": vector_field,
                                "k": num_results
                                * 2,  # Get more results to account for weighting
                            },
                        }
                    },
                    {
                        "$project": {
                            "name": 1,
                            "contact_details": 1,
                            "education": 1,
                            "experience": 1,
                            "projects": 1,
                            "total_experience": 1,
                            "skills": 1,
                            "certifications": 1,
                            "original_score": {"$meta": "searchScore"},
                            "weighted_score": {
                                "$multiply": [{"$meta": "searchScore"}, weight]
                            },
                            "search_field": {"$literal": field},
                        }
                    },
                ]

                field_results = list(collection.aggregate(pipeline))
                search_results.extend(field_results)

        # Combine and calculate aggregate scores
        resume_scores = {}
        for result in search_results:
            resume_id = str(result["_id"])
            weighted_score = result["weighted_score"]

            if resume_id not in resume_scores:
                resume_scores[resume_id] = {
                    "resume": result,
                    "score": weighted_score,
                    "matches": [result["search_field"]],
                }
            else:
                resume_scores[resume_id]["score"] += weighted_score
                resume_scores[resume_id]["matches"].append(result["search_field"])

        # Sort by total weighted score
        sorted_results = sorted(
            resume_scores.values(), key=lambda x: x["score"], reverse=True
        )[:num_results]

        # Format final results
        final_results = []
        for item in sorted_results:
            resume = item["resume"]
            resume["score"] = item["score"]
            resume["matched_fields"] = item["matches"]
            final_results.append(format_resume(resume))

        return final_results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Weighted vector search failed: {str(e)}"
        )

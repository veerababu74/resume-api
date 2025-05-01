import re
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from database.client import get_collection
from core.helpers import format_resume
from core.vectorizer import Vectorizer

from .city_search import extract_city

resumes_collection = get_collection()


router = APIRouter(prefix="/manualserach", tags=["final manual Search"])

vectroizer = Vectorizer()


@router.post("/advanced", response_model=List[Dict[str, Any]])
async def advanced_resume_search(
    experience_titles: List[str] = Body(
        ..., description="List of experience titles to search for (mandatory)"
    ),
    skills: Optional[List[str]] = Body(
        None, description="List of skills to search for"
    ),
    min_education: Optional[str] = Body(
        None,
        description="Minimum education level (e.g., '10th', 'Diploma', 'Degree', 'BTech')",
    ),
    min_experience: Optional[str] = Body(
        None, description="Minimum experience in format 'X years Y months'"
    ),
    locations: Optional[List[str]] = Body(
        None, description="List of city names to search in address"
    ),
    limit: int = Body(10, description="Maximum number of results to return"),
    semantic_match: bool = Body(
        True, description="Use semantic matching for experience titles and skills"
    ),
):
    """
    Advanced search endpoint that combines multiple criteria:
    - Experience titles (mandatory)
    - Skills (optional)
    - Minimum education level (optional)
    - Minimum experience duration (optional)
    - Locations/cities (optional)

    Returns ranked results with the best matches first.
    """
    try:
        if not experience_titles:
            raise HTTPException(
                status_code=400, detail="At least one experience title is required"
            )

        # Parse min_experience string to months
        min_experience_months = 0
        if min_experience:
            min_experience_months = parse_experience_to_months(min_experience)

        # Perform semantic search for experience titles
        exp_title_results = []
        exp_title_ids = set()

        if semantic_match:
            # Use vector search for titles
            for title in experience_titles:
                title_vector = vectroizer.generate_embedding(title)
                pipeline = [
                    {
                        "$search": {
                            "index": "vector_search_index",
                            "knnBeta": {
                                "vector": title_vector,
                                "path": "experience_text_vector",
                                "k": 100,  # Get more results than needed for filtering
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

                title_results = list(resumes_collection.aggregate(pipeline))

                # Add results to list if not already included
                for result in title_results:
                    result_id = str(result["_id"])
                    if result_id not in exp_title_ids:
                        exp_title_ids.add(result_id)
                        exp_title_results.append(result)
        else:
            # Use regex search for titles
            title_patterns = [
                re.compile(f".*{re.escape(title)}.*", re.IGNORECASE)
                for title in experience_titles
            ]
            title_query = {
                "$or": [
                    {"experience.title": {"$regex": pattern}}
                    for pattern in title_patterns
                ]
            }
            title_results = list(resumes_collection.find(title_query))

            for result in title_results:
                result_id = str(result["_id"])
                if result_id not in exp_title_ids:
                    exp_title_ids.add(result_id)
                    exp_title_results.append(result)

        # Filter results based on additional criteria
        filtered_results = []

        for resume in exp_title_results:
            # Skip if we've reached the limit
            if len(filtered_results) >= limit:
                break

            match_score = 0
            resume_id = str(resume["_id"])

            # Filter by skills if provided
            if skills and len(skills) > 0:
                resume_skills = resume.get("skills", [])

                if semantic_match:
                    # Calculate semantic similarity for skills
                    resume_skills_text = (
                        ", ".join(resume_skills) if resume_skills else ""
                    )
                    query_skills_text = ", ".join(skills)

                    if resume_skills_text:
                        resume_skills_vector = vectroizer.generate_embedding(
                            resume_skills_text
                        )
                        query_skills_vector = vectroizer.generate_embedding(
                            query_skills_text
                        )

                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            resume_skills_vector, query_skills_vector
                        )
                        if similarity < 0.5:  # Threshold for similarity
                            continue

                        match_score += similarity * 10  # Weight skills match
                else:
                    # Check for exact skills matches
                    skill_matches = sum(
                        1
                        for skill in skills
                        if any(s.lower() == skill.lower() for s in resume_skills)
                    )
                    if skill_matches == 0:
                        continue

                    match_score += (skill_matches / len(skills)) * 10

            # Filter by minimum education if provided
            if min_education:
                # Define education hierarchy
                education_levels = {
                    "10th": 1,
                    "ssc": 1,
                    "12th": 2,
                    "hsc": 2,
                    "inter": 2,
                    "intermediate": 2,
                    "diploma": 3,
                    "associate": 3,
                    "bachelor": 4,
                    "degree": 4,
                    "btech": 4,
                    "be": 4,
                    "bsc": 4,
                    "ba": 4,
                    "bcom": 4,
                    "master": 5,
                    "mtech": 5,
                    "me": 5,
                    "msc": 5,
                    "ma": 5,
                    "mcom": 5,
                    "mba": 5,
                    "phd": 6,
                    "doctorate": 6,
                }

                min_level = education_levels.get(min_education.lower(), 0)
                if min_level == 0:
                    # If education level not recognized, skip this filter
                    pass
                else:
                    # Extract highest education level from resume
                    highest_level = 0
                    for edu in resume.get("education", []):
                        degree = edu.get("degree", "").lower()
                        # Check each word in the degree
                        for word in degree.split():
                            if word in education_levels:
                                level = education_levels[word]
                                highest_level = max(highest_level, level)
                                break

                    if highest_level < min_level:
                        continue

                    match_score += highest_level

            # Filter by minimum experience if provided
            if min_experience_months > 0:
                # Try to parse total_experience field
                total_exp = resume.get("total_experience", "")
                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(total_exp)

                    if resume_exp_months < min_experience_months:
                        continue

                    # Add more weight to experience that closely matches the requirement
                    if resume_exp_months <= min_experience_months * 1.5:
                        match_score += 10
                    else:
                        match_score += 5
                else:
                    # Calculate from individual experiences if total not available
                    total_months = 0
                    for exp in resume.get("experience", []):
                        duration = exp.get("duration", "")
                        if duration and duration != "N/A":
                            exp_months = parse_experience_to_months(duration)
                            total_months += exp_months

                    if total_months < min_experience_months:
                        continue

                    # Add score based on experience match
                    if total_months <= min_experience_months * 1.5:
                        match_score += 10
                    else:
                        match_score += 5

            # Filter by locations if provided
            if locations and len(locations) > 0:
                address = resume.get("contact_details", {}).get("address", "")
                if address and address != "N/A":
                    extracted_city = extract_city(address)

                    if not extracted_city or not any(
                        location.lower() in extracted_city.lower()
                        or extracted_city.lower() in location.lower()
                        for location in locations
                    ):
                        # No location match, but don't exclude - just don't add score
                        pass
                    else:
                        match_score += 5  # Bonus for location match

            # Add base score from experience title match if available
            if "score" in resume:
                match_score += resume["score"]

            # Add to filtered results with score
            formatted_resume = format_resume(resume)
            formatted_resume["match_score"] = match_score
            filtered_results.append(formatted_resume)

        # Sort by match score
        sorted_results = sorted(
            filtered_results, key=lambda x: x.get("match_score", 0), reverse=True
        )

        # Return top results up to limit
        return sorted_results[:limit]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")


@router.post("/vector-search", response_model=List[Dict[str, Any]])
async def vector_search_resumes(
    experience_titles: List[str] = Body(
        ..., description="List of experience titles to search for (mandatory)"
    ),
    skills: Optional[List[str]] = Body(
        None, description="List of skills to search for"
    ),
    min_education: Optional[str] = Body(
        None,
        description="Minimum education level (e.g., '10th', 'Diploma', 'Degree', 'BTech')",
    ),
    min_experience: Optional[str] = Body(
        None, description="Minimum experience in format 'X years Y months'"
    ),
    locations: Optional[List[str]] = Body(
        None, description="List of city names to search in address"
    ),
    limit: int = Body(10, description="Maximum number of results to return"),
    semantic_match: bool = Body(
        True, description="Use semantic matching for experience titles and skills"
    ),
):
    """
    Vector search endpoint that combines multiple criteria using semantic search.
    """
    try:
        if not experience_titles:
            raise HTTPException(
                status_code=400, detail="At least one experience title is required"
            )

        # Convert experience titles and skills to a single search query
        search_components = []
        search_components.extend(experience_titles)
        if skills:
            search_components.extend(skills)

        search_text = " ".join(search_components)
        search_vector = vectroizer.generate_embedding(search_text)

        # Parse min_experience if provided
        min_experience_months = 0
        if min_experience:
            min_experience_months = parse_experience_to_months(min_experience)

        # Perform vector search
        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": search_vector,
                        "path": "total_resume_vector",
                        "k": limit * 2,  # Get more results for filtering
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
                    "skills": 1,
                    "total_experience": 1,
                    "certifications": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        results = list(resumes_collection.aggregate(pipeline))

        # Filter and score results based on criteria
        filtered_results = []
        for resume in results:
            match_score = resume.get("score", 0)

            # Apply filters similar to advanced_resume_search
            # Education filter
            if min_education:
                education_levels = {
                    "10th": 1,
                    "ssc": 1,
                    "12th": 2,
                    "hsc": 2,
                    "diploma": 3,
                    "degree": 4,
                    "btech": 4,
                    "be": 4,
                    "master": 5,
                    "mtech": 5,
                    "phd": 6,
                }
                min_level = education_levels.get(min_education.lower(), 0)
                highest_level = 0
                for edu in resume.get("education", []):
                    degree = edu.get("degree", "").lower()
                    for word in degree.split():
                        if word in education_levels:
                            level = education_levels[word]
                            highest_level = max(highest_level, level)
                if highest_level < min_level:
                    continue

            # Experience filter
            if min_experience_months > 0:
                total_exp = resume.get("total_experience", "")
                if total_exp and total_exp != "N/A":
                    resume_exp_months = parse_experience_to_months(total_exp)
                    if resume_exp_months < min_experience_months:
                        continue

            # Location filter
            if locations:
                address = resume.get("contact_details", {}).get("address", "")
                if address and address != "N/A":
                    extracted_city = extract_city(address)
                    if not extracted_city or not any(
                        location.lower() in extracted_city.lower()
                        or extracted_city.lower() in location.lower()
                        for location in locations
                    ):
                        continue

            # Add to filtered results
            formatted_resume = format_resume(resume)
            formatted_resume["match_score"] = match_score
            filtered_results.append(formatted_resume)

        # Sort by match score and limit results
        sorted_results = sorted(
            filtered_results, key=lambda x: x.get("match_score", 0), reverse=True
        )[:limit]

        return sorted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


# Helper function to parse experience strings to months
def parse_experience_to_months(experience_str):
    """
    Parse experience string (e.g., "2 years 6 months", "3 years", "8 months") to total months.

    Args:
        experience_str (str): Experience duration string

    Returns:
        int: Total duration in months
    """
    if not experience_str:
        return 0

    # Convert to lowercase for consistent matching
    exp_lower = experience_str.lower()

    # Pattern for years and months
    years_pattern = re.search(r"(\d+)(?:\s*)(year|years|yr|yrs)", exp_lower)
    months_pattern = re.search(r"(\d+)(?:\s*)(month|months|mo|mos)", exp_lower)

    total_months = 0

    # Add years (converted to months)
    if years_pattern:
        years = int(years_pattern.group(1))
        total_months += years * 12

    # Add months
    if months_pattern:
        months = int(months_pattern.group(1))
        total_months += months

    return total_months


# Helper function for calculating cosine similarity between vectors
def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector1, vector2: Vector embeddings

    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to numpy arrays if they aren't already
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # Calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

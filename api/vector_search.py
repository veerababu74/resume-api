# resume_api/api/search.py
from fastapi import APIRouter, Body, Depends
from models.search import VectorSearchQuery
from core.vectorizer import Vectorizer
from database.client import get_collection
from core.helpers import format_resume
import pymongo
from typing import List, Dict, Optional
import re

router = APIRouter(prefix="/vector-search", tags=["Vector Search"])

collection = get_collection()
vectorizer = Vectorizer()


@router.post("/", response_model=List[Dict])
async def vector_search(search_query: VectorSearchQuery):
    try:
        query_embedding = vectorizer.generate_embedding(search_query.query)

        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",
            "projects": "projects_text_vector",
        }

        vector_field = vector_field_mapping.get(search_query.field)
        if not vector_field:
            raise ValueError(
                f"Invalid field name. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": vector_field,
                        "k": search_query.num_results,
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
        return [format_resume(result) for result in results]

    except Exception as e:
        raise Exception(f"Vector search failed: {str(e)}")


@router.post("/hybrid/", response_model=List[Dict])
async def hybrid_search(
    query: str = Body(...),
    fields: List[str] = Body(["skills", "experience", "education", "projects"]),
    keywords: Optional[List[str]] = Body(None),
    num_results: int = Body(5),
):
    try:
        query_embedding = vectorizer.generate_embedding(query)

        vector_field_mapping = {
            "skills": "skills_vector",
            "experience": "experience_text_vector",
            "education": "education_text_vector",
            "projects": "projects_text_vector",
        }

        valid_fields = [f for f in fields if f in vector_field_mapping]
        if not valid_fields:
            raise ValueError(
                f"No valid fields provided. Choose from: {', '.join(vector_field_mapping.keys())}"
            )

        search_results = []
        for field in valid_fields:
            vector_field = vector_field_mapping[field]

            pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": vector_field,
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
                        "search_field": {"$literal": field},
                    }
                },
            ]

            if keywords:
                keyword_conditions = []
                for keyword in keywords:
                    keyword_pattern = re.compile(keyword, re.IGNORECASE)

                    if field == "skills":
                        keyword_conditions.append(
                            {"skills": {"$in": [keyword_pattern]}}
                        )
                    elif field == "experience":
                        keyword_conditions.append(
                            {
                                "$or": [
                                    {"experience.title": {"$regex": keyword_pattern}},
                                    {"experience.company": {"$regex": keyword_pattern}},
                                ]
                            }
                        )
                    elif field == "education":
                        keyword_conditions.append(
                            {
                                "$or": [
                                    {"education.degree": {"$regex": keyword_pattern}},
                                    {
                                        "education.institution": {
                                            "$regex": keyword_pattern
                                        }
                                    },
                                ]
                            }
                        )
                    elif field == "projects":
                        keyword_conditions.append(
                            {
                                "$or": [
                                    {"projects.name": {"$regex": keyword_pattern}},
                                    {
                                        "projects.description": {
                                            "$regex": keyword_pattern
                                        }
                                    },
                                    {
                                        "projects.technologies": {
                                            "$in": [keyword_pattern]
                                        }
                                    },
                                ]
                            }
                        )

                if keyword_conditions:
                    pipeline.insert(1, {"$match": {"$or": keyword_conditions}})

            field_results = list(collection.aggregate(pipeline))
            search_results.extend(field_results)

        seen_ids = set()
        unique_results = []
        for result in sorted(search_results, key=lambda x: x["score"], reverse=True):
            result_id = str(result["_id"])
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
                if len(unique_results) >= num_results:
                    break

        return [format_resume(result) for result in unique_results]

    except Exception as e:
        raise Exception(f"Hybrid search failed: {str(e)}")


@router.post("/experience-title/")
async def experience_title_search(query: str = Body(...), num_results: int = Body(5)):
    try:
        query_embedding = vectorizer.generate_embedding(query)

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "experience_text_vector",
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
        return [format_resume(result) for result in results]

    except Exception as e:
        raise Exception(f"Experience title search failed: {str(e)}")


@router.post("/skills/")
async def skills_search(query: str = Body(...), num_results: int = Body(5)):
    try:
        query_embedding = vectorizer.generate_embedding(query)

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "skills_vector",
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
        return [format_resume(result) for result in results]

    except Exception as e:
        raise Exception(f"Skills search failed: {str(e)}")


@router.post("/education/")
async def education_search(query: str = Body(...), num_results: int = Body(5)):
    try:
        query_embedding = vectorizer.generate_embedding(query)

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "education_text_vector",
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
        return [format_resume(result) for result in results]

    except Exception as e:
        raise Exception(f"Education search failed: {str(e)}")


@router.post("/projects/")
async def projects_search(query: str = Body(...), num_results: int = Body(5)):
    try:
        query_embedding = vectorizer.generate_embedding(query)

        pipeline = [
            {
                "$search": {
                    "index": "vector_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "projects_text_vector",
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
        return [format_resume(result) for result in results]

    except Exception as e:
        raise Exception(f"Projects search failed: {str(e)}")

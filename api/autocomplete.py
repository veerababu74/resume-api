# resume_api/api/autocomplete.py
from fastapi import APIRouter, Query, Body, Depends
from database.client import get_collection
from core.vectorizer import Vectorizer
from core.helpers import format_resume
import pymongo
import re
from typing import List, Dict

router = APIRouter(prefix="/autocomplete", tags=["Autocomplete"])

collection = get_collection()
vectorizer = Vectorizer()


@router.get("/titles/", response_model=List[str])
async def autocomplete_titles(prefix: str = Query(...), limit: int = Query(10)):
    try:
        pipeline = [
            {"$unwind": "$experience"},
            {
                "$match": {
                    "experience.title": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$experience.title"}},
            {"$limit": limit},
            {"$project": {"title": "$_id", "_id": 0}},
        ]

        results = list(collection.aggregate(pipeline))
        titles = [doc["title"] for doc in results]

        if len(titles) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "experience_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$experience"},
                {"$project": {"title": "$experience.title"}},
                {"$limit": limit},
            ]

            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_titles = [
                doc["title"] for doc in semantic_results if doc["title"] not in titles
            ]
            titles.extend(semantic_titles[: limit - len(titles)])

        return titles

    except Exception as e:
        raise Exception(f"Title autocomplete failed: {str(e)}")


@router.get("/skills/", response_model=List[str])
async def autocomplete_skills(prefix: str = Query(...), limit: int = Query(10)):
    try:
        pipeline = [
            {"$unwind": "$skills"},
            {"$match": {"skills": {"$regex": f".*{prefix}.*", "$options": "i"}}},
            {"$group": {"_id": "$skills"}},
            {"$limit": limit},
            {"$project": {"skill": "$_id", "_id": 0}},
        ]

        results = list(collection.aggregate(pipeline))
        skills = [doc["skill"] for doc in results]

        if len(skills) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "skills_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$skills"},
                {"$project": {"skill": "$skills"}},
                {"$limit": limit},
            ]

            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_skills = [
                doc["skill"] for doc in semantic_results if doc["skill"] not in skills
            ]
            skills.extend(semantic_skills[: limit - len(skills)])

        return skills

    except Exception as e:
        raise Exception(f"Skills autocomplete failed: {str(e)}")


@router.get("/degrees/", response_model=List[str])
async def autocomplete_degrees(prefix: str = Query(...), limit: int = Query(10)):
    try:
        pipeline = [
            {"$unwind": "$education"},
            {
                "$match": {
                    "education.degree": {"$regex": f".*{prefix}.*", "$options": "i"}
                }
            },
            {"$group": {"_id": "$education.degree"}},
            {"$limit": limit},
            {"$project": {"degree": "$_id", "_id": 0}},
        ]

        results = list(collection.aggregate(pipeline))
        degrees = [doc["degree"] for doc in results]

        if len(degrees) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "education_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$education"},
                {"$project": {"degree": "$education.degree"}},
                {"$limit": limit},
            ]

            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_degrees = [
                doc["degree"]
                for doc in semantic_results
                if doc["degree"] not in degrees
            ]
            degrees.extend(semantic_degrees[: limit - len(degrees)])

        return degrees

    except Exception as e:
        raise Exception(f"Degree autocomplete failed: {str(e)}")


@router.get("/technologies/", response_model=List[str])
async def autocomplete_technologies(prefix: str = Query(...), limit: int = Query(10)):
    try:
        pipeline = [
            {"$unwind": "$projects"},
            {"$unwind": "$projects.technologies"},
            {
                "$match": {
                    "projects.technologies": {
                        "$regex": f".*{prefix}.*",
                        "$options": "i",
                    }
                }
            },
            {"$group": {"_id": "$projects.technologies"}},
            {"$limit": limit},
            {"$project": {"technology": "$_id", "_id": 0}},
        ]

        results = list(collection.aggregate(pipeline))
        technologies = [doc["technology"] for doc in results]

        if len(technologies) < limit:
            query_embedding = vectorizer.generate_embedding(prefix)

            semantic_pipeline = [
                {
                    "$search": {
                        "index": "vector_search_index",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "projects_text_vector",
                            "k": limit,
                        },
                    }
                },
                {"$unwind": "$projects"},
                {"$unwind": "$projects.technologies"},
                {"$project": {"technology": "$projects.technologies"}},
                {"$limit": limit},
            ]

            semantic_results = list(collection.aggregate(semantic_pipeline))
            semantic_technologies = [
                doc["technology"]
                for doc in semantic_results
                if doc["technology"] not in technologies
            ]
            technologies.extend(semantic_technologies[: limit - len(technologies)])

        return technologies

    except Exception as e:
        raise Exception(f"Technology autocomplete failed: {str(e)}")

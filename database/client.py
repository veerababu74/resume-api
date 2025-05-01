# resume_api/database/client.py
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from config import MONGODB_URI
from fastapi import HTTPException


def get_client():
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi("1"),
            tlsAllowInvalidCertificates=True,
        )
        return client
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to MongoDB: {str(e)}"
        )


def get_collection():
    client = get_client()
    db = client["resume_db"]
    return db["resumes"]

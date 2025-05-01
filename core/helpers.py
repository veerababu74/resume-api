# resume_api/core/helpers.py
from bson import ObjectId
import datetime
import json
import re


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


def format_resume(resume):
    """Convert MongoDB document to serializable format"""
    if resume:
        resume["_id"] = str(resume["_id"])
        # Remove vector fields from response
        for key in list(resume.keys()):
            if key.endswith("_vector"):
                del resume[key]
    return resume


def extract_city(address):
    """Extract city from address string"""
    if not address:
        return None
    # Try to match city pattern
    city_match = re.search(
        r"(?<=,\s)[^,]+(?=,|$)|(?<=\s)[A-Z][a-z]+(?=,\s[A-Z]{2})", address
    )
    if city_match:
        return city_match.group(0)
    # Fallback: Split by comma
    parts = address.split(",")
    if len(parts) > 1:
        return parts[1].strip()
    return None

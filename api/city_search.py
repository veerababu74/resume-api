# resume_api/api/city_search.py
from fastapi import APIRouter
from typing import List
import json
import re
from fastapi import Body, Query, HTTPException
from core.helpers import format_resume
from typing import Dict, Any
from database.client import get_collection

router = APIRouter(prefix="/search", tags=["City Search"])
# Initialize database collection
collection = get_collection()
with open("data/indiancities.json", "r") as f:
    cities = [city.strip() for city in json.load(f)]


@router.get("/indian_cities/", response_model=List[str])
async def autocomplete_cities(q: str, limit: int = 10):
    q_lower = q.lower()
    filtered_cities = [city for city in cities if city.lower().startswith(q_lower)]
    return filtered_cities[:limit]


@router.post("/search/city/", response_model=List[Dict[str, Any]])
async def search_by_city(
    city: str = Body(..., description="City name to search for in address"),
    limit: int = Body(10, description="Maximum number of results to return"),
):
    """
    Search for resumes where the contact address contains the specified city name.
    Returns all resumes that match the city name in the address field.
    """
    try:
        # Case-insensitive search for city name in address field
        city_pattern = re.compile(f"\\b{re.escape(city)}\\b", re.IGNORECASE)

        # Search resumes where address contains the city name
        query = {"contact_details.address": {"$regex": city_pattern}}

        # Get matching resumes
        results = list(collection.find(query).limit(limit))

        # Format results
        formatted_results = [format_resume(result) for result in results]

        # If no exact matches found, try to use the extract_city function to match on cities
        if not formatted_results:
            # Get all resumes
            all_resumes = list(
                collection.find({}).limit(100)
            )  # Limit to prevent too many processing

            # Filter resumes where extracted city matches the search city
            matching_resumes = []
            for resume in all_resumes:
                address = resume.get("contact_details", {}).get("address", "")
                extracted_city = extract_city(address)

                if extracted_city and city_pattern.search(extracted_city):
                    matching_resumes.append(resume)

                if len(matching_resumes) >= limit:
                    break

            # Format matches
            formatted_results = [format_resume(result) for result in matching_resumes]

        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")


# Additional API to get a list of all cities from resume addresses
@router.get("/cities/", response_model=List[str])
async def list_cities(
    limit: int = Query(100, description="Maximum number of cities to return")
):
    """
    Extract and return a list of all distinct cities from resume addresses.
    """
    try:
        # Get all resumes with addresses
        resumes = list(
            collection.find(
                {"contact_details.address": {"$exists": True, "$ne": "N/A"}},
                {"contact_details.address": 1},
            )
        )

        # Extract cities
        cities = set()
        for resume in resumes:
            address = resume.get("contact_details", {}).get("address", "")
            extracted_city = extract_city(address)
            if extracted_city:
                cities.add(extracted_city)

            if len(cities) >= limit:
                break

        return sorted(list(cities))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City listing failed: {str(e)}")


# Improved extract_city function with better pattern matching
def extract_city(address):
    """
    Extract city name from an address string using multiple patterns and approaches.

    Args:
        address (str): Address string containing city name

    Returns:
        str or None: Extracted city name or None if not found
    """
    if not address or address == "N/A":
        return None

    # Pattern for "City, State" or "City, ST" format
    city_state_pattern = re.search(
        r"([A-Za-z\s]+),\s*([A-Za-z]{2}|[A-Za-z\s]+)", address
    )
    if city_state_pattern:
        return city_state_pattern.group(1).strip()

    # Pattern for city after numeric (e.g., "123 Main St, City")
    after_street_pattern = re.search(r"\d+[^,]+,\s*([A-Za-z\s]+)", address)
    if after_street_pattern:
        return after_street_pattern.group(1).strip()

    # Pattern for Indian addresses: typically City appears before PIN code
    indian_pattern = re.search(r"([A-Za-z\s]+)[\s-]*\d{6}", address)
    if indian_pattern:
        return indian_pattern.group(1).strip()

    # Fallback: Split by comma and take second part (if exists)
    parts = address.split(",")
    if len(parts) > 1:
        return parts[1].strip()

    # Last resort: check if address matches any city in the Indian cities list
    for city in cities:  # Using the indiancities.json loaded at startup
        if re.search(r"\b" + re.escape(city) + r"\b", address, re.IGNORECASE):
            return city

    return None

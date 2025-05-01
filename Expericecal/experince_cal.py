import re
import json


def calculate_total_experience(parsed_data):
    total_years = 0.0
    total_months = 0.0
    pattern = re.compile(
        r"(?i)(\d+\.?\d*)\s*(?:years?|yrs?)\b|(\d+\.?\d*)\s*(?:months?|mos?)\b"
    )

    for experience in parsed_data.get("experience", []):
        duration = experience.get("duration", "")
        for match in pattern.finditer(duration):
            years_str, months_str = match.groups()
            if years_str:
                total_years += float(years_str)
            if months_str:
                total_months += float(months_str)

    # Convert all to months and then to years and remaining months
    total_months_all = total_years * 12 + total_months
    years = int(total_months_all // 12)
    months = int(round(total_months_all % 12))

    return f"{years} years, {months} months"


# Example usage with the provided JSON data:
# Load your JSON data and iterate through each entry
# for entry in your_json_data:
#     total_exp = calculate_total_experience(entry["parsed_data"])
#     print(f"{entry['file_name']}: {total_exp}")

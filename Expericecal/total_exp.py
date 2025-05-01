from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re


class ExperienceCalculator:
    def __init__(self):
        self.date_formats = [
            "%B %Y",
            "%b %Y",
            "%m/%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%Y",
            "%m-%Y",
            "%B %d, %Y",
            "%d %B %Y",
            "%d-%m-%Y",
            "%b %d,%Y",
            "%m/%y",
        ]

    def calculate_experience(self, data: Dict[str, Any]) -> Tuple[int, int]:
        total_years = 0
        total_months = 0

        if not data or "experience" not in data:
            return 0, 0

        for exp in data["experience"]:
            years, months = self._process_experience_entry(exp)
            total_years += years
            total_months += months

        total_years += total_months // 12
        total_months = total_months % 12

        return total_years, total_months

    def _process_experience_entry(self, exp: Dict[str, Any]) -> Tuple[int, int]:
        methods = [
            self._calculate_from_duration,
            self._calculate_from_dates,
            self._calculate_from_relative_time,
        ]

        for method in methods:
            years, months = method(exp)
            if years > 0 or months > 0:
                return years, months

        return 0, 0

    def _calculate_from_duration(self, exp: Dict[str, Any]) -> Tuple[int, int]:
        duration = exp.get("duration", "")
        if not isinstance(duration, str):
            return 0, 0

        duration = duration.lower()
        if "present" in duration or "current" in duration:
            return 0, 0

        years = 0
        months = 0

        year_match = re.search(r"(\d+)\s*(?:year|yr|y)", duration)
        month_match = re.search(r"(\d+)\s*(?:month|mon|m)", duration)

        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))

        return years, months

    def _calculate_from_dates(self, exp: Dict[str, Any]) -> Tuple[int, int]:
        start_date = self._parse_date(exp.get("start_date", ""))
        end_date = self._parse_date(exp.get("end_date", ""))

        if not start_date:
            return 0, 0

        if not end_date or exp.get("end_date", "").lower() in ["present", "current"]:
            end_date = datetime.now()

        if start_date and end_date:
            diff = relativedelta(end_date, start_date)
            return diff.years, diff.months

        return 0, 0

    def _calculate_from_relative_time(self, exp: Dict[str, Any]) -> Tuple[int, int]:
        start = exp.get("start_date", "")
        if not isinstance(start, str):
            return 0, 0

        if "ago" in start.lower():
            number = re.search(r"(\d+)", start)
            if not number:
                return 0, 0

            amount = int(number.group(1))
            if "year" in start.lower():
                return amount, 0
            elif "month" in start.lower():
                return 0, amount

        return 0, 0

    def _parse_date(self, date_string: str) -> datetime:
        if not date_string or not isinstance(date_string, str):
            return None

        date_string = date_string.strip()
        if date_string.lower() in ["present", "current", "now"]:
            return datetime.now()

        for fmt in self.date_formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        return None


def format_experience(years: int, months: int) -> str:
    if years == 0 and months == 0:
        return "No experience"

    parts = []
    if years > 0:
        parts.append(f"{years} {'year' if years == 1 else 'years'}")
    if months > 0:
        parts.append(f"{months} {'month' if months == 1 else 'months'}")

    return " and ".join(parts)


calculator = ExperienceCalculator()

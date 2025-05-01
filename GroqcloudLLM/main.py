import os
import json
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# ===== Pydantic Models =====
class Project(BaseModel):
    name: str = Field(default="", description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: List[str] = Field(
        default_factory=list, description="Technologies used"
    )
    role: str = Field(default="", description="Role in the project")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="", description="End date")
    duration: str = Field(default="", description="Total duration")


class ContactDetails(BaseModel):
    email: str = Field(
        default="placeholder@example.com", description="Candidate's email address"
    )
    phone: str = Field(default="+1 123-456-7890", description="Contact phone number")
    address: str = Field(
        default="123 Placeholder St, Placeholder City", description="Physical address"
    )
    linkedin: str = Field(
        default="https://www.linkedin.com/in/placeholder",
        description="LinkedIn profile URL",
    )


class Education(BaseModel):
    degree: str = Field(default="", description="Degree or qualification")
    institution: str = Field(default="", description="Educational institution name")
    dates: str = Field(default="", description="Start and end dates")


class Experience(BaseModel):
    title: str = Field(default="", description="Job position title")
    company: str = Field(default="", description="Company/organization name")
    start_date: str = Field(default="", description="Employment start date")
    end_date: str = Field(default="", description="End date or 'Present'")
    duration: str = Field(default="", description="Total duration in role")


class Resume(BaseModel):
    name: str = Field(default="John Doe", description="Full name of candidate")
    contact_details: ContactDetails
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    total_experience: str = Field(
        default="0 years, 0 months", description="Total work experience duration"
    )
    skills: List[str] = Field(
        default_factory=list, description="List of technical/professional skills"
    )


# ===== Resume Parser Class =====
class ResumeParser:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one API key must be provided.")
        self.api_keys = api_keys
        self.api_usage = {key: 0 for key in api_keys}
        self.current_key_index = 0
        self.processing_chain = self._setup_processing_chain(
            self.api_keys[self.current_key_index]
        )

    def _setup_processing_chain(self, api_key: str):
        os.environ["GROQ_API_KEY"] = api_key
        model = ChatGroq(temperature=1, model_name="gemma2-9b-it")
        parser = JsonOutputParser(pydantic_object=Resume)
        prompt = PromptTemplate(
            template="""Extract resume information strictly as JSON:
{format_instructions}

If experience is present, always calculate total experience.
Ensure the following fields are included: name, email, address, LinkedIn. If any are missing, add random placeholder values.
If there are projects, include them with details like name, description, technologies, role, start_date, end_date, and duration. Use empty values for missing fields.

RESUME INPUT:
{resume_text}

Return ONLY valid JSON without any additional text or explanations.
""",
            input_variables=["resume_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | model | parser

    def _clean_json_response(self, response) -> str:
        if isinstance(response, dict):
            response = json.dumps(response)
        elif not isinstance(response, str):
            response = str(response)

        response = re.sub(
            r"``````", "", response, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        match = re.search(r"({.*})", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response

    def _repair_json(self, malformed_json) -> str:
        repaired = re.sub(r",\s*([}\]])", r"\1", malformed_json)
        repaired = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", repaired)
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            repaired += "{" * (close_braces - open_braces)
        return repaired

    def _extract_json_object(self, text) -> str:
        matches = re.findall(r"({.*?})", text, re.DOTALL)
        if matches:
            return max(matches, key=len)
        return None

    def _repair_and_load_json(self, raw_json) -> dict:
        cleaned_json = self._clean_json_response(raw_json)
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            repaired_json = self._repair_json(cleaned_json)
            try:
                return json.loads(repaired_json)
            except Exception:
                extracted_json = self._extract_json_object(cleaned_json)
                if extracted_json:
                    try:
                        return json.loads(extracted_json)
                    except Exception:
                        pass
                return {"error": "Failed to parse JSON", "raw_output": raw_json}

    def _parse_resume(self, resume_text) -> dict:
        raw_output = self.processing_chain.invoke({"resume_text": resume_text})
        return self._repair_and_load_json(raw_output)

    def process_resume(self, resume_text) -> dict:
        while True:
            try:
                parsed_data = self._parse_resume(resume_text)
                self.api_usage[self.api_keys[self.current_key_index]] += 1
                return parsed_data
            except Exception as e:
                error_msg = str(e).lower()
                print(
                    f"Error with API key {self.api_keys[self.current_key_index]}: {error_msg}"
                )

                if any(
                    x in error_msg
                    for x in [
                        "rate limit",
                        "quota exceeded",
                        "too many requests",
                        "organization_restricted",
                    ]
                ):
                    self.current_key_index += 1
                    if self.current_key_index < len(self.api_keys):
                        self.processing_chain = self._setup_processing_chain(
                            self.api_keys[self.current_key_index]
                        )
                        print(
                            f"Switched to new API key: {self.api_keys[self.current_key_index]}"
                        )
                    else:
                        return {
                            "error": "All API keys exhausted",
                            "api_usage": self.api_usage,
                        }
                else:
                    return {"error": "Unexpected error", "details": error_msg}


# ===== Example Usage =====
if __name__ == "__main__":
    api_keys_list = [
        "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",
        # Add more keys as needed
    ]

    resume_texts = [
        """
        John Doe
        Email: john.doe@example.com
        Phone: +1 555-123-4567
        Address: 123 Main St, Springfield
        LinkedIn: https://www.linkedin.com/in/johndoe
        Experience:
          - Software Engineer at ABC Corp (2018-2022)
          - Intern at XYZ Inc (2017-2018)
        Education:
          - B.Sc. in Computer Science, University of Example (2014-2018)
        Skills: Python, Java, SQL
        Projects:
          - Inventory Management System: Developed a web app using Django and React.
        """
    ]

    parser = ResumeParser(api_keys_list)

    for idx, resume_text in enumerate(resume_texts):
        print(f"Processing Resume {idx + 1}:")
        result = parser.process_resume(resume_text)
        print(json.dumps(result, indent=2))

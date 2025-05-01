# resume_api/models/resume.py
from pydantic import BaseModel, Field
from typing import List, Optional


class ContactDetails(BaseModel):
    email: str = Field(default="N/A")
    phone: str = Field(default="N/A")
    address: str = Field(default="N/A")


class Education(BaseModel):
    degree: str = Field(default="N/A")
    institution: str = Field(default="N/A")
    dates: str = Field(default="N/A")


class Experience(BaseModel):
    title: str = Field(default="N/A")
    company: str = Field(default="N/A")
    start_date: str = Field(default="N/A")
    end_date: str = Field(default="N/A")
    duration: str = Field(default="N/A")


class Project(BaseModel):
    name: str = Field(default="N/A")
    description: str = Field(default="N/A")
    technologies: List[str] = Field(default_factory=list)
    role: str = Field(default="N/A")
    start_date: str = Field(default="N/A")
    end_date: str = Field(default="N/A")
    duration: str = Field(default="N/A")


class ResumeBase(BaseModel):
    name: str = Field(default="N/A")
    contact_details: ContactDetails = Field(default_factory=ContactDetails)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    total_experience: str = Field(default="N/A")
    skills: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class ResumeCreate(ResumeBase):
    pass


class ResumeUpdate(ResumeBase):
    pass


class ResumeInDB(ResumeBase):
    id: str = Field(alias="_id", default=None)

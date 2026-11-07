from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator

FIELDS = ("incident_type", "incident_date", "location", "injuries", "medical_treatment", "liability_factors", "missing_information")
MISSING_KEYS = ("incident_date", "location", "injuries", "medical_treatment", "liability_factors")

class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    incident_type: Literal["vehicle_collision", "slip_and_fall", "workplace_injury"]
    incident_date: date | None
    location: str | None
    injuries: list[str]
    medical_treatment: list[str]
    liability_factors: list[str]
    missing_information: list[str]

    @field_validator("missing_information")
    @classmethod
    def check_missing(cls, v):
        if len(v) != len(set(v)) or any(s not in MISSING_KEYS for s in v):
            raise ValueError("invalid or duplicated missing_information field")
        return v

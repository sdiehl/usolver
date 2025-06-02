from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    """Enum for supported variable types in OR-Tools."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    INTERVAL = "interval"


class Variable(BaseModel):
    """Model representing a variable in an OR-Tools problem."""

    name: str
    type: VariableType
    domain: tuple[int, int] | None = None  # For integer variables
    shape: list[int] | None = None  # For array variables
    description: str = ""


class Constraint(BaseModel):
    """Model representing a constraint in an OR-Tools problem."""

    expression: str
    description: str = ""


class ObjectiveType(str, Enum):
    """Enum for optimization objective types."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    FEASIBILITY = "feasibility"  # Just find a feasible solution


class Objective(BaseModel):
    """Model representing an optimization objective."""

    type: ObjectiveType = ObjectiveType.FEASIBILITY
    expression: str | None = None


class Problem(BaseModel):
    """Model representing a complete OR-Tools constraint programming problem."""

    variables: list[Variable]
    constraints: list[Constraint]
    objective: Objective | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Solution(BaseModel):
    """Model representing a solution to an OR-Tools problem."""

    values: dict[str, Any]
    is_feasible: bool
    status: str
    objective_value: float | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)

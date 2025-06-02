from enum import Enum
from typing import Any

from pydantic import BaseModel


class ObjectiveType(str, Enum):
    """Enum for objective types in CVXPY."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CVXPYVariable(BaseModel):
    """Model representing a CVXPY variable."""

    name: str
    shape: int | tuple[int, ...]  # Can be scalar (int) or vector/matrix (tuple)


class CVXPYConstraint(BaseModel):
    """Model representing a CVXPY constraint."""

    expression: str  # String representation of the constraint
    description: str = ""


class CVXPYObjective(BaseModel):
    """Model representing a CVXPY objective."""

    type: ObjectiveType
    expression: str  # String representation of the objective function


class CVXPYProblem(BaseModel):
    """Model representing a complete CVXPY optimization problem."""

    variables: list[CVXPYVariable]
    objective: CVXPYObjective
    constraints: list[CVXPYConstraint]
    parameters: dict[str, Any] = {}  # For A, b, etc.
    description: str = ""


class CVXPYSolution(BaseModel):
    """Model representing a solution to a CVXPY problem."""

    values: dict[str, Any]  # Variable values
    objective_value: float | None
    status: str
    dual_values: dict[int, Any] | None = None  # Constraint index to dual value

from enum import Enum
from typing import Union

import z3
from pydantic import BaseModel

Z3Value = bool | int | float | str
Z3Ref = z3.BoolRef | z3.ArithRef | z3.ExprRef
Z3Sort = Union[z3.BoolSort, z3.IntSort, z3.RealSort, z3.StringSort]  # noqa: UP007


class Z3VariableType(str, Enum):
    """Variable types in Z3."""

    INTEGER = "integer"
    REAL = "real"
    BOOLEAN = "boolean"
    STRING = "string"


class Z3Variable(BaseModel):
    """Typed variable in a Z3 problem."""

    name: str
    type: Z3VariableType


class Z3Constraint(BaseModel):
    """Constraint in a Z3 problem."""

    expression: str  # expression as string (run through eval)
    description: str = ""


class Z3Problem(BaseModel):
    """Complete Z3 constraint satisfaction problem."""

    variables: list[Z3Variable]
    constraints: list[Z3Constraint]
    description: str = ""


class Z3Solution(BaseModel):
    """Solution to a Z3 problem."""

    values: dict[str, Z3Value]
    is_satisfiable: bool
    status: str

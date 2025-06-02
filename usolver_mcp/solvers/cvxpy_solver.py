from typing import Any, Union

import cvxpy as cp
import numpy as np
from returns.result import Failure, Result, Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYProblem,
    CVXPYSolution,
    ObjectiveType,
)

# Type alias for CVXPY expressions
CVXPYExpr = Union[cp.Expression, cp.Constraint, np.ndarray]  # noqa: UP007


def create_variable(name: str, shape: int | tuple[int, ...]) -> cp.Variable:
    """Create a CVXPY variable with the given shape.

    Args:
        name: Name of the variable
        shape: Shape of the variable (int for scalar, tuple for vector/matrix)

    Returns:
        CVXPY variable
    """
    return cp.Variable(shape, name=name)


def parse_expression(
    expr_str: str, variables: dict[str, cp.Variable], params: dict[str, Any]
) -> CVXPYExpr:
    """Parse a CVXPY expression string.

    Args:
        expr_str: String representation of the expression
        variables: Dictionary of variable names to CVXPY variables
        params: Dictionary of parameter names to values

    Returns:
        Parsed CVXPY expression
    """
    # Create a local dictionary with variables and parameters
    local_dict = {
        **variables,
        **params,
        "cp": cp,
        "np": np,
    }

    # Evaluate the expression in the context of the local dictionary
    return eval(expr_str, {"__builtins__": {}}, local_dict)


def solve_cvxpy_problem(problem: CVXPYProblem) -> Result[CVXPYSolution, str]:
    """Solve a CVXPY optimization problem.

    Args:
        problem: The problem definition

    Returns:
        Result containing a CVXPYSolution or an error message
    """
    try:
        # Create variables
        variables: dict[str, cp.Variable] = {}
        for var in problem.variables:
            variables[var.name] = create_variable(var.name, var.shape)

        # Parse objective
        objective_expr = parse_expression(
            problem.objective.expression, variables, problem.parameters
        )
        objective = (
            cp.Minimize(objective_expr)
            if problem.objective.type == ObjectiveType.MINIMIZE
            else cp.Maximize(objective_expr)
        )

        # Parse constraints
        constraints = []
        for _i, constraint in enumerate(problem.constraints):
            constraint_expr = parse_expression(
                constraint.expression, variables, problem.parameters
            )
            constraints.append(constraint_expr)

        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Extract solution
        values = {name: var.value for name, var in variables.items()}
        dual_values = {
            i: constraint.dual_value
            for i, constraint in enumerate(constraints)
            if hasattr(constraint, "dual_value") and constraint.dual_value is not None
        }

        return Success(
            CVXPYSolution(
                values=values,
                objective_value=float(result) if result is not None else None,
                status=prob.status,
                dual_values=dual_values if dual_values else None,
            )
        )
    except Exception as e:
        return Failure(f"Error solving CVXPY problem: {e!s}")

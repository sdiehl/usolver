#!/usr/bin/env python3
import json
import sys
from typing import Any

from fastmcp import FastMCP
from mcp.types import TextContent
from returns.result import Failure, Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYConstraint,
    CVXPYObjective,
    CVXPYProblem,
    CVXPYVariable,
    ObjectiveType,
)
from usolver_mcp.models.highs_models import (
    HiGHSConstraints,
    HiGHSConstraintSense,
    HiGHSObjective,
    HiGHSOptions,
    HiGHSProblem,
    HiGHSProblemSpec,
    HiGHSSense,
    HiGHSVariable,
    HiGHSVariableType,
)
from usolver_mcp.models.ortools_models import (
    Problem as ORToolsProblem,
)
from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.cvxpy_solver import solve_cvxpy_problem
from usolver_mcp.solvers.highs_solver import solve_problem as solve_highs_problem
from usolver_mcp.solvers.ortools_solver import (
    solve_problem as solve_ortools_problem,
)
from usolver_mcp.solvers.z3_solver import solve_problem

app = FastMCP(
    name="usolver",
    version="0.1.0",
    description="A best-effort universal solver interface for MCP",
    dependencies=[
        "z3-solver>=4.14.1.0",
        "pydantic>=2.0.0",
        "returns>=0.20.0",
        "fastmcp>=0.1.0",
        "cvxpy>=1.6.0",
        "ortools<9.12.0",
        "highspy>=1.7.0",
    ],
)


@app.tool("solve_z3")
async def solve_z3(problem: Z3Problem) -> list[TextContent]:
    """Solve a Z3 constraint satisfaction problem.

    Takes a structured problem definition and returns a solution using Z3 solver.
    Handles both satisfiability and optimization problems.

    Args:
        problem: Problem definition containing variables and constraints

    Returns:
        Solution results as TextContent list, including values and satisfiability status
    """
    result = solve_problem(problem)

    match result:
        case Success(solution):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "values": solution.values,
                            "is_satisfiable": solution.is_satisfiable,
                            "status": solution.status,
                        }
                    ),
                )
            ]
        case Failure(error):
            return [TextContent(type="text", text=f"Error solving problem: {error}")]
        case _:
            return [TextContent(type="text", text="Unexpected error in solve_z3")]


@app.tool("solve_z3_simple")
async def solve_z3_simple(
    variables: list[dict[str, str]],
    constraints: list[str],
    description: str = "",
) -> list[TextContent]:
    """Simplified interface for Z3 constraint problems.

    A more direct way to solve Z3 problems without full model structure.
    Just provide variables and constraints as simple lists.

    Args:
        variables: List of dicts with 'name' and 'type' for each variable
        constraints: List of constraint expressions as strings
        description: Optional problem description

    Returns:
        Solution results as TextContent list
    """
    try:
        # Convert to Problem model
        problem_variables = []
        for var in variables:
            if "name" not in var or "type" not in var:
                return [
                    TextContent(
                        type="text",
                        text="Each variable must have 'name' and 'type' fields",
                    )
                ]

            try:
                var_type = Z3VariableType(var["type"])
            except ValueError:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Invalid variable type: {var['type']}. "
                            f"Must be one of: {', '.join([t.value for t in Z3VariableType])}"
                        ),
                    )
                ]

            problem_variables.append(Z3Variable(name=var["name"], type=var_type))

        problem_constraints = [Z3Constraint(expression=expr) for expr in constraints]

        problem = Z3Problem(
            variables=problem_variables,
            constraints=problem_constraints,
            description=description,
        )

        # Solve the problem
        result = solve_problem(problem)

        match result:
            case Success(solution):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "values": solution.values,
                                "is_satisfiable": solution.is_satisfiable,
                                "status": solution.status,
                            }
                        ),
                    )
                ]
            case Failure(error):
                return [
                    TextContent(type="text", text=f"Error solving problem: {error}")
                ]
            case _:
                return [
                    TextContent(type="text", text="Unexpected error in solve_z3_simple")
                ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in solve_z3_simple: {e!s}")]


@app.tool("solve_highs_problem")
async def solve_highs_problem_tool(problem: HiGHSProblem) -> list[TextContent]:
    """Solve a HiGHs linear/mixed-integer programming problem.

    This tool takes a HiGHs optimization problem defined with variables, objective,
    and constraints, and returns a solution if one exists.

    HiGHs is a high-performance linear programming solver that supports:
    - Linear programming (LP)
    - Mixed-integer programming (MIP)
    - Both dense and sparse constraint matrices
    - Various solver algorithms (simplex, interior point, etc.)

    Example problem structure:
    {
        "problem": {
            "sense": "minimize",
            "objective": {
                "linear": [1.0, 2.0, 3.0]
            },
            "variables": [
                {"name": "x1", "lb": 0, "ub": 10, "type": "cont"},
                {"name": "x2", "lb": 0, "ub": null, "type": "cont"},
                {"name": "x3", "lb": 0, "ub": 1, "type": "bin"}
            ],
            "constraints": {
                "dense": [
                    [1, 1, 0],
                    [0, 1, 1]
                ],
                "sense": ["<=", ">="],
                "rhs": [5, 3]
            }
        },
        "options": {
            "time_limit": 60.0,
            "output_flag": false
        }
    }

    Args:
        problem: The HiGHs problem definition with variables, objective, and constraints

    Returns:
        A list of TextContent containing the solution or an error message
    """
    result = solve_highs_problem(problem)

    match result:
        case Success(solution):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": solution.status.value,
                            "objective_value": solution.objective_value,
                            "solution": solution.solution,
                            "dual_solution": solution.dual_solution,
                            "variable_duals": solution.variable_duals,
                        }
                    ),
                )
            ]
        case Failure(error):
            return [TextContent(type="text", text=f"Error solving problem: {error}")]
        case _:
            return [
                TextContent(
                    type="text",
                    text="Unexpected error in solve_highs_problem_tool",
                )
            ]


@app.tool("simple_highs_solver")
async def simple_highs_solver(
    sense: str,
    objective_coeffs: list[float],
    variables: list[dict[str, Any]],
    constraint_matrix: list[list[float]],
    constraint_senses: list[str],
    rhs_values: list[float],
    options: dict[str, Any] | None = None,
    description: str = "",
) -> list[TextContent]:
    """A simplified interface for solving HiGHs linear programming problems.

    This tool provides a more straightforward interface for HiGHs problems,
    without requiring the full HiGHSProblem model structure.

    Args:
        sense: Optimization sense, either "minimize" or "maximize"
        objective_coeffs: List of objective function coefficients
        variables: List of variable definitions with optional bounds and types
        constraint_matrix: 2D list representing the constraint matrix (dense format)
        constraint_senses: List of constraint directions ("<=", ">=", "=")
        rhs_values: List of right-hand side values for constraints
        options: Optional solver options dictionary
        description: Optional description of the problem

    Returns:
        A list of TextContent containing the solution or an error message
    """
    try:
        # Validate sense
        try:
            problem_sense = HiGHSSense(sense)
        except ValueError:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Invalid sense: {sense}. "
                        f"Must be one of: {', '.join([s.value for s in HiGHSSense])}"
                    ),
                )
            ]

        # Create objective
        objective = HiGHSObjective(linear=objective_coeffs)

        # Create variables
        problem_variables = []
        for i, var in enumerate(variables):
            var_name = var.get("name", f"x{i+1}")
            var_lb = var.get("lb", 0.0)
            var_ub = var.get("ub", None)
            var_type_str = var.get("type", "cont")

            try:
                var_type = HiGHSVariableType(var_type_str)
            except ValueError:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Invalid variable type: {var_type_str}. "
                            f"Must be one of: {', '.join([t.value for t in HiGHSVariableType])}"
                        ),
                    )
                ]

            problem_variables.append(
                HiGHSVariable(name=var_name, lb=var_lb, ub=var_ub, type=var_type)
            )

        # Create constraints
        constraint_sense_enums = []
        for sense_str in constraint_senses:
            try:
                constraint_sense_enums.append(HiGHSConstraintSense(sense_str))
            except ValueError:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Invalid constraint sense: {sense_str}. "
                            f"Must be one of: {', '.join([s.value for s in HiGHSConstraintSense])}"
                        ),
                    )
                ]

        constraints = HiGHSConstraints(
            dense=constraint_matrix,
            sparse=None,
            sense=constraint_sense_enums,
            rhs=rhs_values,
        )

        # Create problem specification
        problem_spec = HiGHSProblemSpec(
            sense=problem_sense,
            objective=objective,
            variables=problem_variables,
            constraints=constraints,
        )

        # Create options if provided
        highs_options = None
        if options:
            highs_options = HiGHSOptions(**options)

        # Create full problem
        problem = HiGHSProblem(problem=problem_spec, options=highs_options)

        # Solve the problem
        result = solve_highs_problem(problem)

        match result:
            case Success(solution):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "status": solution.status.value,
                                "objective_value": solution.objective_value,
                                "solution": solution.solution,
                                "dual_solution": solution.dual_solution,
                                "variable_duals": solution.variable_duals,
                            }
                        ),
                    )
                ]
            case Failure(error):
                return [
                    TextContent(type="text", text=f"Error solving problem: {error}")
                ]
            case _:
                return [
                    TextContent(
                        type="text",
                        text="Unexpected error in simple_highs_solver",
                    )
                ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in simple_highs_solver: {e!s}")]


@app.tool("solve_cvxpy_problem")
async def solve_cvxpy_problem_tool(problem: CVXPYProblem) -> list[TextContent]:
    """Solve a CVXPY optimization problem.

    This tool takes a CVXPY optimization problem defined with variables, objective,
    and constraints, and returns a solution if one exists.

    Example:

    Solve the following problem:

        minimize ||Ax - b||₂²
        subject to:
        0 ≤ x ≤ 1
        where A = [1.0, -0.5; 0.5, 2.0; 0.0, 1.0] and b = [2.0, 1.0, -1.0]

    Should be this tool call:

        simple_cvxpy_solver(
            variables=[{"name": "x", "shape": 2}],
            objective_type="minimize",
            objective_expr="cp.sum_squares(np.array(A) @ x - np.array(b))",
            constraints=["x >= 0", "x <= 1"],
            parameters={"A": [[1.0, -0.5], [0.5, 2.0], [0.0, 1.0]],
                        "b": [2.0, 1.0, -1.0]}
        )

    Args:
        problem: The problem definition with variables, objective, and constraints

    Returns:
        A list of TextContent containing the solution or an error message
    """
    result = solve_cvxpy_problem(problem)

    match result:
        case Success(solution):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "values": {
                                k: v.tolist() if hasattr(v, "tolist") else v
                                for k, v in solution.values.items()
                            },
                            "objective_value": solution.objective_value,
                            "status": solution.status,
                            "dual_values": {
                                k: v.tolist() if hasattr(v, "tolist") else v
                                for k, v in (solution.dual_values or {}).items()
                            },
                        }
                    ),
                )
            ]
        case Failure(error):
            return [TextContent(type="text", text=f"Error solving problem: {error}")]
        case _:
            return [
                TextContent(
                    type="text",
                    text="Unexpected error in solve_cvxpy_problem_tool",
                )
            ]


@app.tool("simple_cvxpy_solver")
async def simple_cvxpy_solver(
    variables: list[dict[str, Any]],
    objective_type: str,
    objective_expr: str,
    constraints: list[str],
    parameters: dict[str, Any] | None = None,
    description: str = "",
) -> list[TextContent]:
    """A simpler interface for solving CVXPY optimization problems.

    This tool provides a more straightforward interface for CVXPY problems,
    without requiring the full CVXPYProblem model structure.

    Args:
        variables: List of variable definitions, each with 'name' and 'shape'
        objective_type: Either 'minimize' or 'maximize'
        objective_expr: The objective function expression as a string
        constraints: List of constraint expressions as strings
        parameters: Dictionary of parameter values (e.g., matrices A, b)
        description: Optional description of the problem

    Returns:
        A list of TextContent containing the solution or an error message
    """
    try:
        # Convert to Problem model
        problem_variables = []
        for var in variables:
            if "name" not in var or "shape" not in var:
                return [
                    TextContent(
                        type="text",
                        text="Each variable must have 'name' and 'shape' fields",
                    )
                ]

            problem_variables.append(CVXPYVariable(**var))

        try:
            obj_type = ObjectiveType(objective_type)
        except ValueError:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Invalid objective type: {objective_type}. "
                        f"Must be one of: {', '.join([t.value for t in ObjectiveType])}"
                    ),
                )
            ]

        objective = CVXPYObjective(type=obj_type, expression=objective_expr)
        problem_constraints = [CVXPYConstraint(expression=expr) for expr in constraints]

        problem = CVXPYProblem(
            variables=problem_variables,
            objective=objective,
            constraints=problem_constraints,
            parameters=parameters or {},
            description=description,
        )

        # Solve the problem
        result = solve_cvxpy_problem(problem)

        match result:
            case Success(solution):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "values": {
                                    k: v.tolist() if hasattr(v, "tolist") else v
                                    for k, v in solution.values.items()
                                },
                                "objective_value": solution.objective_value,
                                "status": solution.status,
                                "dual_values": {
                                    k: v.tolist() if hasattr(v, "tolist") else v
                                    for k, v in (solution.dual_values or {}).items()
                                },
                            }
                        ),
                    )
                ]
            case Failure(error):
                return [
                    TextContent(type="text", text=f"Error solving problem: {error}")
                ]
            case _:
                return [
                    TextContent(
                        type="text",
                        text="Unexpected error in simple_cvxpy_solver",
                    )
                ]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in simple_cvxpy_solver: {e!s}")]


@app.tool("solve_ortools_problem")
async def solve_ortools_problem_tool(
    problem: ORToolsProblem,
) -> list[TextContent]:
    """Solve a constraint programming problem using Google OR-Tools.

    This tool takes a constraint programming problem defined with variables,
    constraints, and an optional objective, and returns a solution if one exists.

    Important Note:
        Each constraint expression must be a single evaluable Python statement.
        You cannot use Python control flow (loops, if statements) in the expressions.
        Instead, you need to generate separate constraints for each case.

    Example:
        Nurse Scheduling Problem:
        ```python
        # Schedule 4 nurses across 3 shifts over 3 days
        shifts_var = Variable(
            name="shifts_var",
            type=VariableType.BOOLEAN,
            shape=[4, 3, 3],  # [nurses, days, shifts]
            description="Binary variable indicating if a nurse works a shift",
        )

        constraints = []

        # INCORRECT - This will fail:
        # Constraint(
        #     expression=(
        #         "for d in range(3): for s in range(3): "
        #         "model.add(sum([shifts_var[n][d][s] for n in range(4)]) == 1)"
        #     )
        # )

        # CORRECT - Add each constraint separately:
        # Each shift must have exactly one nurse
        for d in range(3):
            for s in range(3):
                constraints.append(
                    Constraint(
                        expression=f"model.add(sum([shifts_var[n][{d}][{s}] for n in range(4)]) == 1)",
                        description=f"One nurse for day {d}, shift {s}",
                    )
                )

        # Each nurse works at most one shift per day
        for n in range(4):
            for d in range(3):
                constraints.append(
                    Constraint(
                        expression=f"model.add(sum([shifts_var[{n}][{d}][s] for s in range(3)]) <= 1)",
                        description=f"Max one shift for nurse {n} on day {d}",
                    )
                )

        # Each nurse works 2-3 shifts total
        for n in range(4):
            constraints.append(
                Constraint(
                    expression=f"model.add(sum([shifts_var[{n}][d][s] for d in range(3) for s in range(3)]) >= 2)",
                    description=f"Min shifts for nurse {n}",
                )
            )

        problem = Problem(
            variables=[shifts_var],
            constraints=constraints,
            description="Hospital nurse scheduling problem",
        )
        ```

    Args:
        problem: The problem definition with variables, constraints, and optional objective

    Returns:
        A list of TextContent containing the solution or an error message
    """
    result = solve_ortools_problem(problem)

    match result:
        case Success(solution):
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "values": solution.values,
                            "is_feasible": solution.is_feasible,
                            "status": solution.status,
                            "objective_value": solution.objective_value,
                            "statistics": solution.statistics,
                        }
                    ),
                )
            ]
        case Failure(error):
            return [TextContent(type="text", text=f"Error solving problem: {error}")]
        case _:
            return [
                TextContent(
                    type="text",
                    text="Unexpected error in solve_ortools_problem_tool",
                )
            ]


def main() -> None:
    print("Starting usolver MCP server...", file=sys.stderr)
    app.run()


if __name__ == "__main__":
    main()

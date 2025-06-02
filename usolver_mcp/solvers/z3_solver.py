from typing import cast

import z3
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Ref,
    Z3Solution,
    Z3Value,
    Z3Variable,
    Z3VariableType,
)


def create_variable(variable: Z3Variable) -> Result[tuple[str, Z3Ref], str]:
    """Takes a Variable model and converts it into a Z3 variable representation.

    Args:
        variable: The Variable model to convert.

    Returns:
        Result: A Result containing either a successful tuple of (variable_name, z3_variable)
            or a failure with error details.
    """
    try:
        name = variable.name
        match variable.type:
            case Z3VariableType.INTEGER:
                return Success((name, cast(Z3Ref, z3.Int(name))))
            case Z3VariableType.REAL:
                return Success((name, cast(Z3Ref, z3.Real(name))))
            case Z3VariableType.BOOLEAN:
                return Success((name, cast(Z3Ref, z3.Bool(name))))
            case Z3VariableType.STRING:
                return Success((name, cast(Z3Ref, z3.String(name))))
            case _:
                return Failure(f"Unsupported variable type: {variable.type}")
    except Exception as e:
        return Failure(f"Error creating variable {variable.name}: {e!s}")


def create_variables(
    variables: list[Z3Variable],
) -> Result[dict[str, Z3Ref], str]:
    """Processes a list of variable definitions and creates corresponding Z3 variables.

    Args:
        variables: List of Variable definitions to process.

    Returns:
        Result: A Result containing either a dictionary mapping variable names to Z3 variables
            or a failure with error details.
    """
    result_dict = {}

    for var in variables:
        result = create_variable(var)
        match result:
            case Success(value):
                name, z3_var = value
                result_dict[name] = z3_var
            case Failure(_):
                return result

    return Success(result_dict)


def parse_constraint(
    constraint: Z3Constraint, variables: dict[str, Z3Ref]
) -> Result[z3.BoolRef, str]:
    """Converts a constraint expression into a Z3 constraint using the provided variables.

    Args:
        constraint: The constraint definition to parse.
        variables: Dictionary of variable names to Z3 variables.

    Returns:
        Result: A Result containing either a Z3 boolean reference or a failure with error details.
    """
    try:
        local_dict = {
            **variables,
            **{
                "And": z3.And,
                "Or": z3.Or,
                "Not": z3.Not,
                "Implies": z3.Implies,
                "ForAll": z3.ForAll,
                "Exists": z3.Exists,
                "If": z3.If,
                "Distinct": z3.Distinct,
                "true": True,
                "false": False,
            },
        }

        z3_constraint = eval(constraint.expression, {"__builtins__": {}}, local_dict)
        return Success(z3_constraint)
    except Exception as e:
        return Failure(f"Error parsing constraint '{constraint.expression}': {e!s}")


def create_constraints(
    constraints: list[Z3Constraint], variables: dict[str, Z3Ref]
) -> Result[list[z3.BoolRef], str]:
    """Transforms a list of constraint definitions into Z3 constraints.

    Args:
        constraints: List of constraint definitions to transform.
        variables: Dictionary of variable names to Z3 variables.

    Returns:
        Result: A Result containing either a list of Z3 boolean references or a failure with error details.
    """
    z3_constraints = []

    for constraint in constraints:
        result = parse_constraint(constraint, variables)
        match result:
            case Success(value):
                z3_constraints.append(value)
            case Failure(_):
                return result

    return Success(z3_constraints)


def solve(
    variables: dict[str, Z3Ref], constraints: list[z3.BoolRef]
) -> Result[tuple[z3.CheckSatResult, z3.ModelRef | None], str]:
    """Attempts to solve a Z3 problem with the given variables and constraints.

    Args:
        variables: Dictionary of variable names to Z3 variables.
        constraints: List of Z3 constraints to solve.

    Returns:
        Result: A Result containing either a tuple of (satisfiability_result, model)
            or a failure with error details.
    """
    try:
        solver = z3.Solver()
        solver.add(constraints)

        result = solver.check()
        model = None

        if result == z3.sat:
            model = solver.model()

        return Success((result, model))
    except Exception as e:
        return Failure(f"Error solving constraints: {e!s}")


def get_z3_value(model: z3.ModelRef, z3_var: Z3Ref) -> Maybe[Z3Value]:
    """Extracts a value from a Z3 model for a given variable using Maybe monad.

    Args:
        model: The Z3 model to extract from.
        z3_var: The Z3 variable to extract.

    Returns:
        Maybe: A Maybe containing either the extracted value or Nothing if extraction fails.
    """
    try:
        z3_value = model[z3_var]
        if z3_value is None:
            return Nothing

        str_value = str(z3_value)

        match z3_var:
            case _ if z3.is_bool(z3_var):
                return Some(z3.is_true(z3_value))
            case _ if z3.is_int(z3_var):
                return Some(int(str_value))
            case _ if z3.is_real(z3_var):
                if "/" in str_value:
                    num_str, den_str = str_value.split("/")
                    return Some(float(int(num_str)) / float(int(den_str)))
                else:
                    return Some(float(str_value))
            case _:
                return Some(str_value)
    except Exception:
        return Nothing


def extract_solution(
    result: z3.CheckSatResult,
    model: z3.ModelRef | None,
    variables: dict[str, Z3Ref],
) -> Result[Z3Solution, str]:
    """Processes a Z3 solution and extracts variable values from the model.

    Args:
        result: The satisfiability result from Z3.
        model: The Z3 model if satisfiable, None otherwise.
        variables: Dictionary of variable names to Z3 variables.

    Returns:
        Result: A Result containing either a Z3Solution or a failure with error details.
    """
    try:
        match result:
            case z3.sat:
                if model is None:
                    return Failure("Model is None but result is sat")
                values = {
                    name: maybe_value.unwrap()
                    for name, z3_var in variables.items()
                    for maybe_value in [get_z3_value(model, z3_var)]
                    if isinstance(maybe_value, Some)
                }
                return Success(
                    Z3Solution(
                        values=values,
                        is_satisfiable=True,
                        status=str(result),
                    )
                )
            case _:
                return Success(
                    Z3Solution(
                        values={},
                        is_satisfiable=False,
                        status=str(result),
                    )
                )
    except Exception as e:
        return Failure(f"Error extracting solution: {e!s}")


def solve_problem(problem: Z3Problem) -> Result[Z3Solution, str]:
    """Orchestrates the complete process of solving a Z3 problem.

    Args:
        problem: The problem definition containing variables and constraints.

    Returns:
        Result: A Result containing either a Solution object or a failure with error details.
    """
    try:
        vars_result = create_variables(problem.variables)
        if isinstance(vars_result, Failure):
            return vars_result

        variables = vars_result.unwrap()

        constraints_result = create_constraints(problem.constraints, variables)
        if isinstance(constraints_result, Failure):
            return constraints_result

        z3_constraints = constraints_result.unwrap()

        solve_result = solve(variables, z3_constraints)
        if isinstance(solve_result, Failure):
            return solve_result

        result_tuple = solve_result.unwrap()

        return extract_solution(result_tuple[0], result_tuple[1], variables)
    except Exception as e:
        return Failure(f"Error solving problem: {e!s}")

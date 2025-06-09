import highspy
import numpy as np
from returns.result import Failure, Result, Success

from usolver_mcp.models.highs_models import (
    HiGHSConstraintSense,
    HiGHSOptions,
    HiGHSOutput,
    HiGHSProblem,
    HiGHSProblemSpec,
    HiGHSSense,
    HiGHSStatus,
    HiGHSVariable,
    HiGHSVariableType,
)


def _convert_sense_to_highs(sense: HiGHSSense) -> int:
    """Convert HiGHSSense to HiGHs minimize flag."""
    return 1 if sense == HiGHSSense.MINIMIZE else -1


def _convert_constraint_sense(
    sense: HiGHSConstraintSense, rhs: float
) -> tuple[float, float]:
    """Convert constraint sense to lower and upper bounds."""
    inf = highspy.kHighsInf

    match sense:
        case HiGHSConstraintSense.LESS_EQUAL:
            return (-inf, rhs)
        case HiGHSConstraintSense.GREATER_EQUAL:
            return (rhs, inf)
        case HiGHSConstraintSense.EQUAL:
            return (rhs, rhs)


def _get_variable_bounds(
    var_spec: HiGHSVariable, var_index: int
) -> tuple[float, float]:
    """Get variable bounds, applying defaults based on variable type."""
    inf = highspy.kHighsInf

    # Default bounds based on type
    if var_spec.type == HiGHSVariableType.BINARY:
        default_lb, default_ub = 0.0, 1.0
    else:
        default_lb, default_ub = 0.0, inf

    lb = var_spec.lb if var_spec.lb is not None else default_lb
    ub = var_spec.ub if var_spec.ub is not None else default_ub

    return (lb, ub)


def _build_constraint_matrix(
    problem_spec: HiGHSProblemSpec,
) -> Result[tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    """Build constraint matrix in sparse format."""
    try:
        constraints = problem_spec.constraints

        if constraints.dense is not None:
            # Convert dense matrix to sparse format
            dense_matrix = np.array(constraints.dense)
            rows, cols = np.nonzero(dense_matrix)
            values = dense_matrix[rows, cols]

            return Success((rows.astype(int), cols.astype(int), values.astype(float)))

        elif constraints.sparse is not None:
            # Use provided sparse format
            sparse = constraints.sparse
            rows = np.array(sparse.rows, dtype=int)
            cols = np.array(sparse.cols, dtype=int)
            values = np.array(sparse.values, dtype=float)

            return Success((rows, cols, values))
        else:
            return Failure("Either dense or sparse constraint matrix must be provided")

    except Exception as e:
        return Failure(f"Error building constraint matrix: {e}")


def _apply_options(
    h: "highspy.Highs", options: HiGHSOptions | None
) -> Result[None, str]:
    """Apply solver options to HiGHs instance."""
    try:
        if options is None:
            return Success(None)

        # Time limit
        if options.time_limit is not None:
            h.setOptionValue("time_limit", options.time_limit)

        # Presolve
        if options.presolve is not None:
            h.setOptionValue("presolve", options.presolve.value)

        # Solver
        if options.solver is not None:
            h.setOptionValue("solver", options.solver.value)

        # Parallel
        if options.parallel is not None:
            h.setOptionValue("parallel", options.parallel.value)

        # Threads
        if options.threads is not None:
            h.setOptionValue("threads", options.threads)

        # Random seed
        if options.random_seed is not None:
            h.setOptionValue("random_seed", options.random_seed)

        # Tolerances
        if options.primal_feasibility_tolerance is not None:
            h.setOptionValue(
                "primal_feasibility_tolerance", options.primal_feasibility_tolerance
            )

        if options.dual_feasibility_tolerance is not None:
            h.setOptionValue(
                "dual_feasibility_tolerance", options.dual_feasibility_tolerance
            )

        # Logging
        if options.output_flag is not None:
            h.setOptionValue("output_flag", options.output_flag)

        if options.log_to_console is not None:
            h.setOptionValue("log_to_console", options.log_to_console)

        return Success(None)

    except Exception as e:
        return Failure(f"Error applying options: {e}")


def _convert_status(model_status: "highspy.HighsModelStatus") -> HiGHSStatus:
    """Convert HiGHs model status to HiGHSStatus enum."""
    # HiGHs status constants
    if model_status == highspy.HighsModelStatus.kOptimal:
        return HiGHSStatus.OPTIMAL
    elif model_status == highspy.HighsModelStatus.kInfeasible:
        return HiGHSStatus.INFEASIBLE
    elif model_status == highspy.HighsModelStatus.kUnbounded:
        return HiGHSStatus.UNBOUNDED
    elif model_status == highspy.HighsModelStatus.kUnboundedOrInfeasible:
        return HiGHSStatus.UNBOUNDED  # or could be a separate status
    else:
        return HiGHSStatus.UNKNOWN


def solve_problem(problem: HiGHSProblem) -> Result[HiGHSOutput, str]:
    """Solve a HiGHs optimization problem."""
    try:
        # Create HiGHs instance
        h = highspy.Highs()

        # Always disable output for MCP server compatibility
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)

        # Apply options
        options_result = _apply_options(h, problem.options)
        if isinstance(options_result, Failure):
            return options_result

        problem_spec = problem.problem
        num_vars = len(problem_spec.variables)
        num_constraints = len(problem_spec.constraints.sense)

        # Set up objective
        obj_coeffs = np.array(problem_spec.objective.linear, dtype=float)
        if len(obj_coeffs) != num_vars:
            return Failure(
                f"Objective coefficients length ({len(obj_coeffs)}) doesn't match number of variables ({num_vars})"
            )

        # Set up variable bounds
        var_lower = np.zeros(num_vars, dtype=float)
        var_upper = np.full(num_vars, highspy.kHighsInf, dtype=float)

        for i, var_spec in enumerate(problem_spec.variables):
            lb, ub = _get_variable_bounds(var_spec, i)
            var_lower[i] = lb
            var_upper[i] = ub

        # Build constraint matrix
        matrix_result = _build_constraint_matrix(problem_spec)
        if isinstance(matrix_result, Failure):
            return matrix_result

        rows, cols, values = matrix_result.unwrap()

        # Set up constraint bounds
        constraint_lower = np.zeros(num_constraints, dtype=float)
        constraint_upper = np.zeros(num_constraints, dtype=float)

        for i, (sense, rhs) in enumerate(
            zip(
                problem_spec.constraints.sense,
                problem_spec.constraints.rhs,
                strict=False,
            )
        ):
            lb, ub = _convert_constraint_sense(sense, rhs)
            constraint_lower[i] = lb
            constraint_upper[i] = ub

        # Add variables
        h.addCols(
            num_vars,
            obj_coeffs,
            var_lower,
            var_upper,
            0,
            np.array([]),
            np.array([]),
            np.array([]),
        )

        # Set variable integrality constraints
        integrality = np.zeros(num_vars, dtype=int)  # 0 = continuous
        for i, var_spec in enumerate(problem_spec.variables):
            if var_spec.type == HiGHSVariableType.BINARY:
                integrality[i] = 1  # 1 = integer (binary is integer with bounds 0-1)
            elif var_spec.type == HiGHSVariableType.INTEGER:
                integrality[i] = 1  # 1 = integer
            # else: continuous (already 0)
        
        # Apply integrality constraints if any variables are integer/binary
        if np.any(integrality > 0):
            h.changeColsIntegrality(num_vars, np.arange(num_vars), integrality)

        # Add constraints using sparse format
        if len(rows) > 0:
            # Convert to row-wise sparse format for HiGHs
            # Group by rows and create start array
            unique_rows = np.unique(rows)
            start_array = np.zeros(num_constraints + 1, dtype=int)

            for row in unique_rows:
                start_array[row] = np.sum(rows < row)

            start_array[-1] = len(rows)  # Final start

            h.addRows(
                num_constraints,
                constraint_lower,
                constraint_upper,
                len(values),
                start_array,
                cols,
                values,
            )
        else:
            # No constraints case
            h.addRows(
                num_constraints,
                constraint_lower,
                constraint_upper,
                0,
                np.array([0]),
                np.array([]),
                np.array([]),
            )

        # Set objective sense
        if problem_spec.sense == HiGHSSense.MAXIMIZE:
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            h.changeObjectiveSense(highspy.ObjSense.kMinimize)

        # Solve the problem
        h.run()

        # Get results
        model_status = h.getModelStatus()
        solution = h.getSolution()
        info = h.getInfo()

        # Convert status
        status = _convert_status(model_status)

        # Extract solution values
        solution_values = solution.col_value if hasattr(solution, "col_value") else []
        dual_values = solution.row_dual if hasattr(solution, "row_dual") else []
        reduced_costs = solution.col_dual if hasattr(solution, "col_dual") else []

        # Get objective value
        objective_value = (
            info.objective_function_value
            if hasattr(info, "objective_function_value")
            else 0.0
        )

        return Success(
            HiGHSOutput(
                status=status,
                objective_value=objective_value,
                solution=list(solution_values),
                dual_solution=list(dual_values),
                variable_duals=list(reduced_costs),
            )
        )

    except Exception as e:
        return Failure(f"Error solving HiGHs problem: {e}")


def simple_highs_solver(
    sense: str,
    objective_coeffs: list[float],
    variables: list[dict],
    constraint_matrix: list[list[float]],
    constraint_senses: list[str],
    rhs_values: list[float],
    options: dict | None = None,
    description: str = "",
) -> Result[HiGHSOutput, str]:
    """A simplified interface for solving HiGHs linear programming problems.

    This function provides a more straightforward interface for HiGHs problems,
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
        Result containing HiGHSOutput or error message
    """
    try:
        from usolver_mcp.models.highs_models import (
            HiGHSConstraints,
            HiGHSObjective,
            HiGHSProblem,
            HiGHSProblemSpec,
        )

        # Validate sense
        try:
            problem_sense = HiGHSSense(sense)
        except ValueError:
            return Failure(
                f"Invalid sense: {sense}. "
                f"Must be one of: {', '.join([s.value for s in HiGHSSense])}"
            )

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
                return Failure(
                    f"Invalid variable type: {var_type_str}. "
                    f"Must be one of: {', '.join([t.value for t in HiGHSVariableType])}"
                )

            problem_variables.append(
                HiGHSVariable(name=var_name, lb=var_lb, ub=var_ub, type=var_type)
            )

        # Create constraints
        constraint_sense_enums = []
        for sense_str in constraint_senses:
            try:
                constraint_sense_enums.append(HiGHSConstraintSense(sense_str))
            except ValueError:
                return Failure(
                    f"Invalid constraint sense: {sense_str}. "
                    f"Must be one of: {', '.join([s.value for s in HiGHSConstraintSense])}"
                )

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
        return solve_problem(problem)

    except Exception as e:
        return Failure(f"Error in simple_highs_solver: {e!s}")

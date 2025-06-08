from returns.result import Success

from usolver_mcp.models.highs_models import (
    HiGHSConstraints,
    HiGHSConstraintSense,
    HiGHSObjective,
    HiGHSOptions,
    HiGHSProblem,
    HiGHSProblemSpec,
    HiGHSSense,
    HiGHSSparseMatrix,
    HiGHSStatus,
    HiGHSVariable,
    HiGHSVariableType,
)
from usolver_mcp.solvers.highs_solver import solve_problem

pytest_plugins: list[str] = []


def test_simple_linear_program() -> None:
    """Test solving a simple linear program.

    Minimize    f  =  x0 +  x1
    subject to              x1 <= 7
                5 <=  x0 + 2x1 <= 15
                6 <= 3x0 + 2x1
                0 <= x0 <= 4; 1 <= x1
    """
    # Define variables
    variables = [
        HiGHSVariable(name="x0", lb=0, ub=4, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="x1", lb=1, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    # Define objective: minimize x0 + x1
    objective = HiGHSObjective(linear=[1.0, 1.0])

    # Define constraints using dense format
    # x1 <= 7  =>  [0, 1] * [x0, x1] <= 7
    # 5 <= x0 + 2*x1 <= 15  =>  [1, 2] * [x0, x1] >= 5 and [1, 2] * [x0, x1] <= 15
    # 6 <= 3*x0 + 2*x1  =>  [3, 2] * [x0, x1] >= 6
    constraints = HiGHSConstraints(
        dense=[
            [0, 1],  # x1 <= 7
            [1, 2],  # x0 + 2*x1 >= 5
            [1, 2],  # x0 + 2*x1 <= 15
            [3, 2],  # 3*x0 + 2*x1 >= 6
        ],
        sparse=None,
        sense=[
            HiGHSConstraintSense.LESS_EQUAL,  # x1 <= 7
            HiGHSConstraintSense.GREATER_EQUAL,  # x0 + 2*x1 >= 5
            HiGHSConstraintSense.LESS_EQUAL,  # x0 + 2*x1 <= 15
            HiGHSConstraintSense.GREATER_EQUAL,  # 3*x0 + 2*x1 >= 6
        ],
        rhs=[7, 5, 15, 6],
    )

    # Create problem
    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    # Add some options
    options = HiGHSOptions(  # type: ignore[call-arg]
        output_flag=False, log_to_console=False  # Suppress solver output for testing
    )

    problem = HiGHSProblem(problem=problem_spec, options=options)

    # Solve the problem
    result = solve_problem(problem)

    # Check that we got a successful result
    assert isinstance(result, Success)

    solution = result.unwrap()

    # Check that we found an optimal solution
    assert solution.status == HiGHSStatus.OPTIMAL

    # Check that we have solution values for both variables
    assert len(solution.solution) == 2

    # Check that the objective value is reasonable
    # The optimal solution should be around x0=0, x1=2.5 with objective value 2.5
    assert solution.objective_value > 0
    assert solution.objective_value < 10  # Should be a small positive value

    # Check that we have dual information
    assert len(solution.dual_solution) == 4  # One for each constraint
    assert len(solution.variable_duals) == 2  # One for each variable


def test_sparse_constraint_format() -> None:
    """Test solving with sparse constraint format."""
    # Simple problem: minimize x + y subject to x + y >= 1, x >= 0, y >= 0
    variables = [
        HiGHSVariable(name="x", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="y", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    objective = HiGHSObjective(linear=[1.0, 1.0])

    # Use sparse format: x + y >= 1
    sparse_matrix = HiGHSSparseMatrix(
        rows=[0, 0],  # Both coefficients are in row 0
        cols=[0, 1],  # First coefficient is for variable 0, second for variable 1
        values=[1.0, 1.0],  # Both coefficients are 1.0
        shape=(1, 2),  # 1 constraint, 2 variables
    )

    constraints = HiGHSConstraints(
        dense=None,
        sparse=sparse_matrix, 
        sense=[HiGHSConstraintSense.GREATER_EQUAL], 
        rhs=[1.0]
    )

    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    options = HiGHSOptions(output_flag=False, log_to_console=False)  # type: ignore[call-arg]
    problem = HiGHSProblem(problem=problem_spec, options=options)

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()
    assert solution.status == HiGHSStatus.OPTIMAL

    # Optimal solution should be x=0.5, y=0.5 with objective value 1.0
    assert abs(solution.objective_value - 1.0) < 1e-6




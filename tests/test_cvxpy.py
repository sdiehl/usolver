from returns.result import Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYConstraint,
    CVXPYObjective,
    CVXPYProblem,
    CVXPYVariable,
    ObjectiveType,
)
from usolver_mcp.solvers.cvxpy_solver import solve_cvxpy_problem

pytest_plugins: list[str] = []


def test_simple_linear_regression() -> None:
    """Test solving a simple linear regression problem using CVXPY.

    Problem: minimize ||Ax - b||²
    where A = [[1, 1], [2, 1], [3, 1]] and b = [3, 5, 7]
    Expected solution: x ≈ [2, 1] (slope=2, intercept=1)
    """
    problem = CVXPYProblem(
        variables=[CVXPYVariable(name="x", shape=2)],
        objective=CVXPYObjective(
            type=ObjectiveType.MINIMIZE,
            expression="cp.sum_squares(np.array(A) @ x - np.array(b))",
        ),
        constraints=[],
        parameters={"A": [[1, 1], [2, 1], [3, 1]], "b": [3, 5, 7]},
        description="Simple linear regression problem",
    )

    result = solve_cvxpy_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    # Check that the problem was solved successfully
    assert solution.status == "optimal"
    assert solution.objective_value is not None
    assert solution.objective_value < 1e-10  # Should be very close to 0

    # Check solution values (should be approximately [2, 1])
    x_values = solution.values["x"]
    assert abs(x_values[0] - 2.0) < 1e-6  # slope
    assert abs(x_values[1] - 1.0) < 1e-6  # intercept


def test_constrained_optimization() -> None:
    """Test CVXPY with constraints.

    Problem: minimize x² + y²
    subject to: x + y >= 2, x >= 0, y >= 0
    Expected solution: x = 1, y = 1, objective = 2
    """
    problem = CVXPYProblem(
        variables=[CVXPYVariable(name="x", shape=1), CVXPYVariable(name="y", shape=1)],
        objective=CVXPYObjective(
            type=ObjectiveType.MINIMIZE,
            expression="cp.sum_squares(x) + cp.sum_squares(y)",
        ),
        constraints=[
            CVXPYConstraint(expression="x + y >= 2"),
            CVXPYConstraint(expression="x >= 0"),
            CVXPYConstraint(expression="y >= 0"),
        ],
        description="Constrained quadratic optimization",
    )

    result = solve_cvxpy_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    assert solution.status == "optimal"
    assert solution.objective_value is not None

    # Check solution values (should be approximately x=1, y=1)
    x_val = (
        solution.values["x"][0]
        if hasattr(solution.values["x"], "__len__")
        else solution.values["x"]
    )
    y_val = (
        solution.values["y"][0]
        if hasattr(solution.values["y"], "__len__")
        else solution.values["y"]
    )

    assert abs(x_val - 1.0) < 1e-4
    assert abs(y_val - 1.0) < 1e-4
    assert abs(solution.objective_value - 2.0) < 1e-4


def test_portfolio_optimization() -> None:
    """Test a simple portfolio optimization problem.

    Problem: minimize risk while achieving target return
    Two assets with different risk/return profiles
    """
    problem = CVXPYProblem(
        variables=[CVXPYVariable(name="weights", shape=2)],
        objective=CVXPYObjective(
            type=ObjectiveType.MINIMIZE, expression="cp.quad_form(weights, cov_matrix)"
        ),
        constraints=[
            CVXPYConstraint(expression="cp.sum(weights) == 1"),  # weights sum to 1
            CVXPYConstraint(expression="weights >= 0"),  # long-only
            CVXPYConstraint(
                expression="np.array(returns).T @ weights >= target_return"
            ),  # target return
        ],
        parameters={
            "cov_matrix": [[0.04, 0.01], [0.01, 0.09]],  # covariance matrix
            "returns": [0.08, 0.12],  # expected returns
            "target_return": 0.10,  # target 10% return
        },
        description="Simple portfolio optimization",
    )

    result = solve_cvxpy_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    assert solution.status == "optimal"
    assert solution.objective_value is not None

    # Check that weights sum to 1
    weights = solution.values["weights"]
    assert abs(sum(weights) - 1.0) < 1e-6

    # Check that all weights are non-negative
    assert all(w >= -1e-6 for w in weights)


def test_infeasible_problem() -> None:
    """Test CVXPY with an infeasible problem.

    Problem: minimize x
    subject to: x >= 2, x <= 1
    This should be infeasible.
    """
    problem = CVXPYProblem(
        variables=[CVXPYVariable(name="x", shape=1)],
        objective=CVXPYObjective(type=ObjectiveType.MINIMIZE, expression="x"),
        constraints=[
            CVXPYConstraint(expression="x >= 2"),
            CVXPYConstraint(expression="x <= 1"),
        ],
        description="Infeasible optimization problem",
    )

    result = solve_cvxpy_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    # Should be infeasible
    assert solution.status == "infeasible"
    assert solution.objective_value is None or solution.objective_value == float("inf")

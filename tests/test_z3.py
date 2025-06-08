from returns.result import Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.z3_solver import solve_problem

pytest_plugins: list[str] = []


def test_worker_task_assignment_problem() -> None:
    """Test solving a worker-task assignment problem using Z3.

    Problem Summary:
    A simple scheduling problem where two workers (A and B) need to be assigned
    to two tasks (1 and 2) with the following constraints:

    - Each worker can only be assigned to one task
    - Each task must be assigned to exactly one worker
    - Worker A prefers task 1 (soft constraint/preference)

    This demonstrates Z3's ability to handle boolean satisfiability problems
    with logical constraints and preferences.
    """
    # Define the problem
    problem = Z3Problem(
        variables=[
            # Binary variables for worker assignments
            Z3Variable(
                name="A1", type=Z3VariableType.BOOLEAN
            ),  # Worker A assigned to Task 1
            Z3Variable(
                name="A2", type=Z3VariableType.BOOLEAN
            ),  # Worker A assigned to Task 2
            Z3Variable(
                name="B1", type=Z3VariableType.BOOLEAN
            ),  # Worker B assigned to Task 1
            Z3Variable(
                name="B2", type=Z3VariableType.BOOLEAN
            ),  # Worker B assigned to Task 2
        ],
        constraints=[
            # Worker A can only be assigned to one task
            Z3Constraint(
                expression="And(Or(A1, A2), Not(And(A1, A2)))",
                description="Worker A must be assigned to exactly one task",
            ),
            # Worker B can only be assigned to one task
            Z3Constraint(
                expression="And(Or(B1, B2), Not(And(B1, B2)))",
                description="Worker B must be assigned to exactly one task",
            ),
            # Task 1 must be assigned to one worker
            Z3Constraint(
                expression="And(Or(A1, B1), Not(And(A1, B1)))",
                description="Task 1 must be assigned to exactly one worker",
            ),
            # Task 2 must be assigned to one worker
            Z3Constraint(
                expression="And(Or(A2, B2), Not(And(A2, B2)))",
                description="Task 2 must be assigned to exactly one worker",
            ),
            # Preference: Worker A prefers task 1 (soft constraint)
            Z3Constraint(
                expression="A1",
                description="Preference for Worker A to be assigned to Task 1",
            ),
        ],
        description="Worker-Task Assignment Problem with Preferences",
    )

    # Solve the problem
    result = solve_problem(problem)

    # Verify the solution
    assert isinstance(result, Success)
    solution = result.unwrap()

    # Check that the problem is satisfiable
    assert solution.is_satisfiable
    assert solution.status == "sat"

    # Check that we have values for all 4 variables
    assert len(solution.values) == 4
    assert all(var in solution.values for var in ["A1", "A2", "B1", "B2"])

    # Verify that all values are boolean
    for var_name, value in solution.values.items():
        assert isinstance(
            value, bool
        ), f"Variable {var_name} should be boolean, got {type(value)}"

    # Check constraint satisfaction
    values = solution.values

    # Worker A assigned to exactly one task
    a_assignments = sum([values["A1"], values["A2"]])
    assert a_assignments == 1, "Worker A must be assigned to exactly one task"

    # Worker B assigned to exactly one task
    b_assignments = sum([values["B1"], values["B2"]])
    assert b_assignments == 1, "Worker B must be assigned to exactly one task"

    # Task 1 assigned to exactly one worker
    task1_assignments = sum([values["A1"], values["B1"]])
    assert task1_assignments == 1, "Task 1 must be assigned to exactly one worker"

    # Task 2 assigned to exactly one worker
    task2_assignments = sum([values["A2"], values["B2"]])
    assert task2_assignments == 1, "Task 2 must be assigned to exactly one worker"

    # Check preference: Worker A should be assigned to Task 1 (due to preference constraint)
    assert (
        values["A1"] is True
    ), "Worker A should be assigned to Task 1 due to preference"
    assert values["B2"] is True, "Worker B should be assigned to Task 2"
    assert values["A2"] is False, "Worker A should not be assigned to Task 2"
    assert values["B1"] is False, "Worker B should not be assigned to Task 1"


def test_simple_arithmetic_constraints() -> None:
    """Test Z3 with simple arithmetic constraints.

    Problem: Find integer values for x and y such that:
    - x + y = 10
    - x - y = 2
    - x > 0, y > 0

    Expected solution: x = 6, y = 4
    """
    problem = Z3Problem(
        variables=[
            Z3Variable(name="x", type=Z3VariableType.INTEGER),
            Z3Variable(name="y", type=Z3VariableType.INTEGER),
        ],
        constraints=[
            Z3Constraint(expression="x + y == 10", description="Sum equals 10"),
            Z3Constraint(expression="x - y == 2", description="Difference equals 2"),
            Z3Constraint(expression="x > 0", description="x is positive"),
            Z3Constraint(expression="y > 0", description="y is positive"),
        ],
        description="Simple arithmetic constraint problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()
    assert solution.is_satisfiable

    # Check the expected solution
    assert solution.values["x"] == 6
    assert solution.values["y"] == 4

    # Verify constraints are satisfied
    x, y = solution.values["x"], solution.values["y"]
    assert x + y == 10
    assert x - y == 2
    assert x > 0
    assert y > 0


def test_unsatisfiable_problem() -> None:
    """Test Z3 with an unsatisfiable problem.

    Problem: Find a boolean variable x such that:
    - x is true
    - x is false

    This should be unsatisfiable.
    """
    problem = Z3Problem(
        variables=[
            Z3Variable(name="x", type=Z3VariableType.BOOLEAN),
        ],
        constraints=[
            Z3Constraint(expression="x", description="x must be true"),
            Z3Constraint(expression="Not(x)", description="x must be false"),
        ],
        description="Unsatisfiable boolean problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    # Should be unsatisfiable
    assert not solution.is_satisfiable
    assert solution.status == "unsat"
    assert len(solution.values) == 0  # No solution values for unsatisfiable problems

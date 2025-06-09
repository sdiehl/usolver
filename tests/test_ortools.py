from returns.result import Success

from usolver_mcp.models.ortools_models import (
    Constraint,
    Objective,
    ObjectiveType,
    Problem,
    Variable,
    VariableType,
)
from usolver_mcp.solvers.ortools_solver import solve_problem

pytest_plugins: list[str] = []


def test_simple_boolean_satisfiability() -> None:
    """Test solving a simple boolean satisfiability problem.

    Problem: Find boolean values for x, y, z such that:
    - x OR y = True
    - y OR z = True
    - NOT x OR NOT z = True

    One valid solution: x=True, y=False, z=True
    """
    problem = Problem(
        variables=[
            Variable(name="x", type=VariableType.BOOLEAN),
            Variable(name="y", type=VariableType.BOOLEAN),
            Variable(name="z", type=VariableType.BOOLEAN),
        ],
        constraints=[
            Constraint(
                expression="model.add_bool_or([x, y])",
                description="x OR y must be true",
            ),
            Constraint(
                expression="model.add_bool_or([y, z])",
                description="y OR z must be true",
            ),
            Constraint(
                expression="model.add_bool_or([x.Not(), z.Not()])",
                description="NOT x OR NOT z must be true",
            ),
        ],
        description="Simple boolean satisfiability problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    # Check that the problem is feasible
    assert solution.is_feasible
    assert solution.status in ["FEASIBLE", "OPTIMAL"]

    # Check that we have values for all variables
    assert len(solution.values) == 3
    assert all(var in solution.values for var in ["x", "y", "z"])

    # Verify constraints are satisfied
    x, y, z = solution.values["x"], solution.values["y"], solution.values["z"]
    assert x or y  # x OR y
    assert y or z  # y OR z
    assert not x or not z  # NOT x OR NOT z


def test_integer_programming() -> None:
    """Test OR-Tools with integer variables.

    Problem: Find integers x, y such that:
    - 2x + 3y <= 14
    - x + y >= 2
    - x >= 0, y >= 0
    - maximize x + 2y

    Expected solution: x=1, y=4 with objective value 9
    """
    problem = Problem(
        variables=[
            Variable(name="x", type=VariableType.INTEGER, domain=(0, 10)),
            Variable(name="y", type=VariableType.INTEGER, domain=(0, 10)),
        ],
        constraints=[
            Constraint(
                expression="model.add(2*x + 3*y <= 14)",
                description="Resource constraint",
            ),
            Constraint(
                expression="model.add(x + y >= 2)", description="Minimum production"
            ),
        ],
        objective=Objective(type=ObjectiveType.MAXIMIZE, expression="x + 2*y"),
        description="Integer programming problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    assert solution.is_feasible
    assert solution.status in ["FEASIBLE", "OPTIMAL"]

    # Check constraint satisfaction
    x, y = solution.values["x"], solution.values["y"]
    assert 2 * x + 3 * y <= 14
    assert x + y >= 2
    assert x >= 0 and y >= 0

    # Check that we found a good solution
    assert solution.objective_value is not None
    assert solution.objective_value >= 8  # Should be close to optimal


def test_simple_scheduling() -> None:
    """Test a simple scheduling problem with OR-Tools.

    Problem: Schedule 2 tasks on 2 time slots
    - Each task must be scheduled exactly once
    - No two tasks can be scheduled at the same time
    """
    problem = Problem(
        variables=[
            Variable(
                name="schedule",
                type=VariableType.BOOLEAN,
                shape=[2, 2],  # [tasks, time_slots]
                description="schedule[i][j] = task i is scheduled at time j",
            ),
        ],
        constraints=[
            # Each task scheduled exactly once
            Constraint(
                expression="model.add(sum([schedule[0][t] for t in range(2)]) == 1)",
                description="Task 0 scheduled exactly once",
            ),
            Constraint(
                expression="model.add(sum([schedule[1][t] for t in range(2)]) == 1)",
                description="Task 1 scheduled exactly once",
            ),
            # No two tasks at same time
            Constraint(
                expression="model.add(sum([schedule[i][0] for i in range(2)]) <= 1)",
                description="At most one task at time 0",
            ),
            Constraint(
                expression="model.add(sum([schedule[i][1] for i in range(2)]) <= 1)",
                description="At most one task at time 1",
            ),
        ],
        description="Simple task scheduling problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    assert solution.is_feasible
    assert solution.status in ["FEASIBLE", "OPTIMAL"]

    # Check solution structure
    schedule = solution.values["schedule"]
    assert len(schedule) == 2  # 2 tasks
    assert len(schedule[0]) == 2  # 2 time slots
    assert len(schedule[1]) == 2  # 2 time slots

    # Verify constraints
    # Each task scheduled exactly once
    assert sum(schedule[0]) == 1
    assert sum(schedule[1]) == 1

    # No conflicts at each time slot
    assert schedule[0][0] + schedule[1][0] <= 1  # time 0
    assert schedule[0][1] + schedule[1][1] <= 1  # time 1


def test_infeasible_problem() -> None:
    """Test OR-Tools with an infeasible problem.

    Problem: Find boolean x such that:
    - x = True
    - x = False

    This should be infeasible.
    """
    problem = Problem(
        variables=[
            Variable(name="x", type=VariableType.BOOLEAN),
        ],
        constraints=[
            Constraint(expression="model.add(x == True)", description="x must be true"),
            Constraint(
                expression="model.add(x == False)", description="x must be false"
            ),
        ],
        description="Infeasible boolean problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    # Should be infeasible
    assert not solution.is_feasible
    assert solution.status in ["INFEASIBLE", "INVALID"]
    assert len(solution.values) == 0  # No solution values


def test_optimization_with_arrays() -> None:
    """Test OR-Tools optimization with array variables.

    Problem: Binary knapsack with 3 items
    - Items have weights [2, 3, 4] and values [3, 4, 5]
    - Knapsack capacity is 5
    - Maximize total value

    Expected: select items 0 and 1 for total value 7
    """
    problem = Problem(
        variables=[
            Variable(
                name="selected",
                type=VariableType.BOOLEAN,
                shape=[3],  # 3 items
                description="selected[i] = item i is selected",
            ),
        ],
        constraints=[
            Constraint(
                expression="model.add(2*selected[0] + 3*selected[1] + 4*selected[2] <= 5)",
                description="Weight capacity constraint",
            ),
        ],
        objective=Objective(
            type=ObjectiveType.MAXIMIZE,
            expression="3*selected[0] + 4*selected[1] + 5*selected[2]",
        ),
        description="Binary knapsack problem",
    )

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()

    assert solution.is_feasible
    assert solution.status in ["FEASIBLE", "OPTIMAL"]

    # Check solution
    selected = solution.values["selected"]
    assert len(selected) == 3

    # Check constraint satisfaction
    total_weight = 2 * selected[0] + 3 * selected[1] + 4 * selected[2]
    assert total_weight <= 5

    # Check that we found a good solution
    assert solution.objective_value is not None
    assert solution.objective_value >= 6  # Should find a decent solution

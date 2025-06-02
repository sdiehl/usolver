#!/usr/bin/env python3
from returns.result import Failure, Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.z3_solver import solve_problem

# Define a simple scheduling problem:
# - Two workers (A and B) need to be assigned to two tasks (1 and 2)
# - Each worker can only be assigned to one task
# - Each task must be assigned to one worker
# - Worker A prefers task 1 (represented by a constraint that encourages this assignment)

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

# Print the result
match result:
    case Success(solution):
        print("Solution status:", solution.status)
        print("\nAssignments:")
        for var_name, value in solution.values.items():
            print(f"  {var_name} = {value}")

        # Interpret the results
        print("\nInterpretation:")
        for var_name, value in solution.values.items():
            if value:
                worker = "Worker A" if var_name.startswith("A") else "Worker B"
                task = "Task 1" if var_name.endswith("1") else "Task 2"
                print(f"  {worker} is assigned to {task}")

    case Failure(error):
        print("Error:", error)

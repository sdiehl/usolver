#!/usr/bin/env python3
import numpy as np
from returns.result import Failure, Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYConstraint,
    CVXPYObjective,
    CVXPYProblem,
    CVXPYVariable,
    ObjectiveType,
)
from usolver_mcp.solvers.cvxpy_solver import solve_cvxpy_problem

# Problem data
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define the optimization problem
problem = CVXPYProblem(
    variables=[CVXPYVariable(name="x", shape=n)],  # n-dimensional vector
    objective=CVXPYObjective(
        type=ObjectiveType.MINIMIZE, expression="cp.sum_squares(A @ x - b)"
    ),
    constraints=[
        CVXPYConstraint(expression="0 <= x"),
        CVXPYConstraint(expression="x <= 1"),
    ],
    parameters={"A": A, "b": b},
)

# Solve the problem
result = solve_cvxpy_problem(problem)

# Print the result
match result:
    case Success(solution):
        print("Solution status:", solution.status)
        print("Objective value:", solution.objective_value)
        print("Variable values:")
        for var_name, value in solution.values.items():
            print(f"  {var_name}:", value)
        if solution.dual_values:
            print("Dual values:")
            for i, value in solution.dual_values.items():
                print(f"  Constraint {i}:", value)
    case Failure(error):
        print("Error:", error)

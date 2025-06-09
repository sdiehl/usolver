"""
Multilinear Optimization Example

This module demonstrates solving a multilinear optimization problem using Z3.
The problem involves minimizing a linear objective function subject to linear constraints
with mixed variable types (continuous and bounded).

Problem formulation:
    minimize: 12x + 20y
    subject to:
        6x + 8y ≥ 100
        7x + 12y ≥ 120
        x ≥ 0
        y ∈ [0, 3]

This demonstrates mixed linear programming using Z3 SMT solving over real numbers
with optimization objectives handled through constraint satisfaction.
"""

from returns.result import Failure, Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.z3_solver import solve_problem


def create_multilinear_problem():
    """
    Create a multilinear optimization problem.

    The problem is:
        minimize: 12x + 20y
        subject to:
            6x + 8y ≥ 100
            7x + 12y ≥ 120
            x ≥ 0
            y ∈ [0, 3]

    Since Z3 is a satisfiability solver rather than an optimizer, we'll use
    a binary search approach to find the minimum value of the objective function.

    Returns:
        tuple: (variables, constraints, objective_bounds) for the Z3 solver
    """
    # Design variables
    variables = [
        {"name": "x", "type": "real"},  # Continuous variable x
        {"name": "y", "type": "real"},  # Continuous variable y
        {"name": "obj", "type": "real"},  # Objective function value
    ]

    constraints = []

    # Variable bounds
    constraints.append("x >= 0")  # x ≥ 0
    constraints.append("y >= 0")  # y ≥ 0 (lower bound)
    constraints.append("y <= 3")  # y ≤ 3 (upper bound)

    # Problem constraints
    constraints.append("6*x + 8*y >= 100")  # First constraint: 6x + 8y ≥ 100
    constraints.append("7*x + 12*y >= 120")  # Second constraint: 7x + 12y ≥ 120

    # Objective function definition
    constraints.append("obj == 12*x + 20*y")  # obj = 12x + 20y

    return variables, constraints


def solve_multilinear_optimization():
    """
    Solve the multilinear optimization problem using binary search.

    Since Z3 is a satisfiability solver, we use binary search to find
    the minimum value of the objective function.

    Returns:
        dict: Solution results including optimal x, y values and minimum objective
    """
    variables, base_constraints = create_multilinear_problem()

    # Convert to Z3Problem model
    z3_variables = [
        Z3Variable(name=var["name"], type=Z3VariableType(var["type"]))
        for var in variables
    ]

    # Binary search for minimum objective value
    # First, find feasible bounds
    lower_bound = 0.0
    upper_bound = 1000.0  # Start with a large upper bound

    # Find a feasible upper bound
    while upper_bound < 10000:
        constraints = [*base_constraints, f"obj <= {upper_bound}"]
        z3_constraints = [
            Z3Constraint(expression=constraint) for constraint in constraints
        ]

        problem = Z3Problem(
            variables=z3_variables,
            constraints=z3_constraints,
            description="Multilinear optimization - finding upper bound",
        )

        result = solve_problem(problem)
        if isinstance(result, Success) and result.unwrap().is_satisfiable:
            break
        upper_bound *= 2

    # Binary search for optimal value
    epsilon = 0.001  # Precision
    best_solution = None

    while upper_bound - lower_bound > epsilon:
        mid = (lower_bound + upper_bound) / 2

        constraints = [*base_constraints, f"obj <= {mid}"]
        z3_constraints = [
            Z3Constraint(expression=constraint) for constraint in constraints
        ]

        problem = Z3Problem(
            variables=z3_variables,
            constraints=z3_constraints,
            description=f"Multilinear optimization - testing obj <= {mid}",
        )

        result = solve_problem(problem)

        if isinstance(result, Success) and result.unwrap().is_satisfiable:
            # Feasible at this objective value
            upper_bound = mid
            solution = result.unwrap()
            best_solution = {
                "x": float(solution.values["x"]),
                "y": float(solution.values["y"]),
                "objective": float(solution.values["obj"]),
            }
        else:
            # Not feasible, increase lower bound
            lower_bound = mid

    if best_solution:
        return {
            "status": "optimal",
            "x": best_solution["x"],
            "y": best_solution["y"],
            "objective": best_solution["objective"],
        }
    else:
        return {"status": "infeasible", "error": "No feasible solution found"}


def solve_multilinear_feasibility():
    """
    Find any feasible solution to verify the problem is solvable.

    Returns:
        dict: A feasible solution if one exists
    """
    variables, constraints = create_multilinear_problem()

    # Convert to Z3Problem model
    z3_variables = [
        Z3Variable(name=var["name"], type=Z3VariableType(var["type"]))
        for var in variables
    ]

    z3_constraints = [Z3Constraint(expression=constraint) for constraint in constraints]

    problem = Z3Problem(
        variables=z3_variables,
        constraints=z3_constraints,
        description="Multilinear optimization - feasibility check",
    )

    result = solve_problem(problem)

    # Parse the result
    match result:
        case Success(solution):
            if solution.is_satisfiable:
                return {
                    "status": "feasible",
                    "x": float(solution.values["x"]),
                    "y": float(solution.values["y"]),
                    "objective": float(solution.values["obj"]),
                }
            else:
                return {"status": "infeasible", "error": "No feasible solution found"}
        case Failure(error):
            return {"status": "error", "error": str(error)}
        case _:
            return {"status": "error", "error": "Unexpected result type"}


def analyze_solution(results):
    """Analyze the optimization solution and verify constraint satisfaction."""
    if results["status"] not in ["optimal", "feasible"]:
        return None

    x = results["x"]
    y = results["y"]
    obj = results["objective"]

    # Verify constraints
    constraint1 = 6 * x + 8 * y  # Should be ≥ 100
    constraint2 = 7 * x + 12 * y  # Should be ≥ 120
    calculated_obj = 12 * x + 20 * y

    # Check constraint violations
    violations = []
    if constraint1 < 100 - 1e-6:
        violations.append(f"Constraint 1: 6x + 8y = {constraint1:.3f} < 100")
    if constraint2 < 120 - 1e-6:
        violations.append(f"Constraint 2: 7x + 12y = {constraint2:.3f} < 120")
    if x < -1e-6:
        violations.append(f"Variable bound: x = {x:.3f} < 0")
    if y < -1e-6:
        violations.append(f"Variable bound: y = {y:.3f} < 0")
    if y > 3 + 1e-6:
        violations.append(f"Variable bound: y = {y:.3f} > 3")

    return {
        "constraint1_value": constraint1,
        "constraint2_value": constraint2,
        "calculated_objective": calculated_obj,
        "objective_error": abs(calculated_obj - obj),
        "violations": violations,
        "is_feasible": len(violations) == 0,
    }


def print_results(results) -> None:
    """Print multilinear optimization results in a formatted way."""
    print("Multilinear Optimization Results")
    print("=" * 50)

    if results["status"] == "error":
        print(f"Error: {results['error']}")
        return
    elif results["status"] == "infeasible":
        print("Problem is infeasible - no solution exists")
        return

    x = results["x"]
    y = results["y"]
    obj = results["objective"]

    print(f"\nStatus: {results['status'].title()}")
    print("\nOptimal Solution:")
    print(f"  x = {x:.6f}")
    print(f"  y = {y:.6f}")
    print(f"  Objective value = {obj:.6f}")

    # Analyze the solution
    analysis = analyze_solution(results)
    if analysis:
        print("\nConstraint Verification:")
        print("-" * 25)
        print(f"  6x + 8y = {analysis['constraint1_value']:.3f} ≥ 100 ✓")
        print(f"  7x + 12y = {analysis['constraint2_value']:.3f} ≥ 120 ✓")
        print(f"  x = {x:.3f} ≥ 0 ✓")
        print(f"  y = {y:.3f} ∈ [0, 3] ✓")

        print("\nObjective Function:")
        print(f"  12x + 20y = {analysis['calculated_objective']:.6f}")
        print(f"  Calculation error: {analysis['objective_error']:.2e}")

        if analysis["violations"]:
            print("\nConstraint Violations:")
            for violation in analysis["violations"]:
                print(f"  ⚠️  {violation}")
        else:
            print("\n✅ All constraints satisfied")

    # Show problem formulation
    print("\nProblem Formulation:")
    print("-" * 20)
    print("  minimize: 12x + 20y")
    print("  subject to:")
    print("    6x + 8y ≥ 100")
    print("    7x + 12y ≥ 120")
    print("    x ≥ 0")
    print("    y ∈ [0, 3]")


def main() -> None:
    """Main function to run the multilinear optimization example."""
    print(__doc__)

    # First check feasibility
    print("Step 1: Checking feasibility...")
    feasible_result = solve_multilinear_feasibility()

    if feasible_result["status"] != "feasible":
        print("Problem is infeasible!")
        print_results(feasible_result)
        return

    print("✅ Problem is feasible")
    print(
        f"Sample feasible point: x={feasible_result['x']:.3f}, "
        f"y={feasible_result['y']:.3f}, obj={feasible_result['objective']:.3f}"
    )

    # Now solve for optimality
    print("\nStep 2: Finding optimal solution...")
    optimal_result = solve_multilinear_optimization()
    print_results(optimal_result)


def test_multilinear_optimization() -> None:
    """Test function for the multilinear optimization example."""
    result = solve_multilinear_optimization()

    assert (
        result["status"] == "optimal"
    ), f"Expected optimal solution, got {result['status']}"

    # Verify the solution satisfies constraints
    x, y = result["x"], result["y"]

    # Check constraints
    assert (
        6 * x + 8 * y >= 100 - 1e-6
    ), f"Constraint 1 violated: 6*{x} + 8*{y} = {6*x + 8*y} < 100"
    assert (
        7 * x + 12 * y >= 120 - 1e-6
    ), f"Constraint 2 violated: 7*{x} + 12*{y} = {7*x + 12*y} < 120"
    assert x >= -1e-6, f"Variable bound violated: x = {x} < 0"
    assert y >= -1e-6, f"Variable bound violated: y = {y} < 0"
    assert y <= 3 + 1e-6, f"Variable bound violated: y = {y} > 3"

    print("✅ All tests passed!")


if __name__ == "__main__":
    main()

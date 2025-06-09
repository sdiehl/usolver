"""
Production Planning Optimization Example

This module demonstrates production planning optimization for a manufacturing facility.
The problem involves determining optimal production quantities for different products
to maximize profit while respecting resource constraints such as machine time,
labor hours, and material availability.

The linear programming problem maximizes total profit subject to:
- Limited machine hours across production lines
- Limited labor hours for assembly and quality control
- Material inventory constraints
- Non-negativity constraints on production quantities
- Optional demand bounds for each product

This is a classic linear programming problem suitable for HiGHS.
"""

from returns.result import Failure, Success

from usolver_mcp.solvers.highs_solver import simple_highs_solver


def create_production_problem():
    """
    Create a production planning optimization problem.

    Returns:
        dict: Problem parameters for the HiGHS solver
    """
    # Products: [ProductA, ProductB, ProductC]

    # Profit per unit for each product
    profit_per_unit = [25.0, 40.0, 35.0]

    # Resource constraints matrix
    # Rows: [Machine Hours, Labor Hours, Material Units]
    # Columns: [ProductA, ProductB, ProductC]
    resource_matrix = [
        [2.0, 3.0, 2.5],  # Machine hours per unit
        [1.0, 2.0, 1.5],  # Labor hours per unit
        [3.0, 1.0, 2.0],  # Material units per unit
    ]

    # Available resources
    available_resources = [
        100.0,  # Machine hours available
        80.0,  # Labor hours available
        120.0,  # Material units available
    ]

    # Variable definitions (all continuous, non-negative)
    variables = [
        {"name": "ProductA", "lb": 0, "type": "cont"},
        {"name": "ProductB", "lb": 0, "type": "cont"},
        {"name": "ProductC", "lb": 0, "type": "cont"},
    ]

    # Constraint senses (all less than or equal to available resources)
    constraint_senses = ["<=", "<=", "<="]

    return {
        "sense": "maximize",
        "objective_coeffs": profit_per_unit,
        "variables": variables,
        "constraint_matrix": resource_matrix,
        "constraint_senses": constraint_senses,
        "rhs_values": available_resources,
        "description": "Production planning optimization to maximize profit",
    }


def solve_production_planning():
    """
    Solve the production planning problem and return results.

    Returns:
        dict: Solution results including optimal production quantities and profit
    """
    problem_params = create_production_problem()
    result = simple_highs_solver(**problem_params)

    # Parse the result from the HiGHS solver
    match result:
        case Success(solution):
            if solution.status.value == "optimal":
                # Extract solution values
                production_quantities = solution.solution or []

                # Ensure we have 3 values for the 3 products
                if len(production_quantities) >= 3:
                    return {
                        "status": "optimal",
                        "production_quantities": production_quantities[:3],
                        "total_profit": solution.objective_value or 0,
                    }
                else:
                    return {"status": "error", "error": "Incomplete solution"}
            else:
                return {
                    "status": solution.status.value,
                    "error": f"Problem status: {solution.status.value}",
                }
        case Failure(error):
            return {"status": "error", "error": str(error)}
        case _:
            return {"status": "error", "error": "Unexpected result type"}


def analyze_solution(results, problem_params):
    """Analyze the solution and calculate resource utilization."""
    if results["status"] != "optimal":
        return None

    production = results["production_quantities"]
    resource_matrix = problem_params["constraint_matrix"]
    available_resources = problem_params["rhs_values"]

    # Calculate resource usage
    resource_usage = []
    for i in range(len(resource_matrix)):
        usage = sum(
            resource_matrix[i][j] * production[j] for j in range(len(production))
        )
        resource_usage.append(usage)

    # Calculate utilization percentages
    utilization = [
        usage / available
        for usage, available in zip(resource_usage, available_resources, strict=False)
    ]

    return {
        "resource_usage": resource_usage,
        "resource_utilization": utilization,
        "available_resources": available_resources,
    }


def print_results(results, problem_params) -> None:
    """Print production planning results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    products = ["Product A", "Product B", "Product C"]
    production = results["production_quantities"]

    print("Production Planning Optimization Results")
    print("=" * 55)
    print("\nOptimal Production Quantities:")
    for i, product in enumerate(products):
        print(f"{product}: {production[i]:8.2f} units")

    print(f"\nMaximum Total Profit: ${results['total_profit']:,.2f}")

    # Resource utilization analysis
    analysis = analyze_solution(results, problem_params)
    if analysis:
        resource_names = ["Machine Hours", "Labor Hours", "Material Units"]

        print("\nResource Utilization:")
        for i, name in enumerate(resource_names):
            usage = analysis["resource_usage"][i]
            available = analysis["available_resources"][i]
            utilization = analysis["resource_utilization"][i]
            print(f"{name:14}: {usage:6.2f} / {available:6.2f} ({utilization:6.1%})")


def main() -> None:
    """Main function to run the production planning example."""
    print(__doc__)

    problem_params = create_production_problem()
    results = solve_production_planning()
    print_results(results, problem_params)


def test_production_planning() -> None:
    """Test function for pytest."""
    problem_params = create_production_problem()
    results = solve_production_planning()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test that production quantities are non-negative
    production = results["production_quantities"]
    assert all(q >= 0 for q in production)

    # Test that resource constraints are satisfied
    resource_matrix = problem_params["constraint_matrix"]
    available_resources = problem_params["rhs_values"]

    for i in range(len(resource_matrix)):
        used = sum(
            resource_matrix[i][j] * production[j] for j in range(len(production))
        )
        assert used <= available_resources[i] + 1e-6  # Allow small numerical errors

    # Test that total profit is positive (assuming feasible problem)
    assert results["total_profit"] >= 0


if __name__ == "__main__":
    main()

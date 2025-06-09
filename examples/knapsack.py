"""
Knapsack Problem Optimization Example

This module demonstrates the classic 0/1 knapsack problem using constraint programming.
The problem involves selecting items to maximize value while staying within a weight
capacity constraint. This is a fundamental combinatorial optimization problem.

The constraint programming problem involves:
- Binary decision variables for each item (take or leave)
- Weight capacity constraint (total weight ≤ capacity)
- Value maximization objective
- Optional: multiple knapsacks or additional constraints

This is a classic integer programming problem suitable for OR-Tools CP-SAT.
"""

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


def create_knapsack_problem(capacity=100, include_advanced_constraints=False):
    """
    Create a knapsack optimization problem.

    Args:
        capacity: Weight capacity of the knapsack
        include_advanced_constraints: Whether to include additional constraints

    Returns:
        Problem: The OR-Tools constraint programming problem
    """
    # Items with (value, weight, name) tuples
    items = [
        (60, 20, "Laptop"),
        (100, 30, "Camera"),
        (120, 40, "Tablet"),
        (80, 25, "Books"),
        (40, 15, "Clothes"),
        (70, 35, "Tools"),
        (90, 45, "Equipment"),
        (50, 20, "Electronics"),
        (110, 50, "Software"),
        (30, 10, "Accessories"),
        (85, 30, "Gadgets"),
        (95, 35, "Supplies"),
    ]

    n_items = len(items)
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    [item[2] for item in items]

    # Decision variables: binary choice for each item
    selected = Variable(
        name="selected",
        type=VariableType.BOOLEAN,
        shape=[n_items],
        description="Binary variable indicating if item i is selected",
    )

    constraints = []

    # Weight capacity constraint
    weight_expr = " + ".join([f"{weights[i]} * selected[{i}]" for i in range(n_items)])
    constraints.append(
        Constraint(
            expression=f"model.add({weight_expr} <= {capacity})",
            description=f"Total weight must not exceed {capacity}",
        )
    )

    # Optional advanced constraints
    if include_advanced_constraints:
        # Constraint: If laptop is selected, tablet must also be selected
        constraints.append(
            Constraint(
                expression="model.add(selected[0] <= selected[2])",  # Laptop -> Tablet
                description="If laptop is selected, tablet must also be selected",
            )
        )

        # Constraint: Can select at most 2 electronics items (laptop, camera, electronics, gadgets)
        electronics_indices = [0, 1, 7, 10]  # Laptop, Camera, Electronics, Gadgets
        electronics_expr = " + ".join([f"selected[{i}]" for i in electronics_indices])
        constraints.append(
            Constraint(
                expression=f"model.add({electronics_expr} <= 2)",
                description="At most 2 electronics items can be selected",
            )
        )

    # Objective: Maximize total value
    value_expr = " + ".join([f"{values[i]} * selected[{i}]" for i in range(n_items)])
    objective = Objective(
        type=ObjectiveType.MAXIMIZE,
        expression=value_expr,
    )

    return Problem(
        variables=[selected],
        constraints=constraints,
        objective=objective,
        description=f"Knapsack problem with capacity {capacity}",
        parameters={
            "capacity": capacity,
            "items": items,
            "enumerate_all_solutions": False,
        },
    )


def solve_knapsack(capacity=100, include_advanced_constraints=False):
    """
    Solve the knapsack problem and return results.

    Args:
        capacity: Weight capacity of the knapsack
        include_advanced_constraints: Whether to include additional constraints

    Returns:
        dict: Solution results including selected items and statistics
    """
    problem = create_knapsack_problem(capacity, include_advanced_constraints)
    result = solve_problem(problem)

    # Item information for analysis
    items = problem.parameters["items"]

    match result:
        case Success(solution):
            if solution.is_feasible:
                selected_items = solution.values["selected"]

                # Analyze the solution
                total_value = 0
                total_weight = 0
                selected_list = []

                for i, is_selected in enumerate(selected_items):
                    if is_selected:
                        value, weight, name = items[i]
                        total_value += value
                        total_weight += weight
                        selected_list.append(
                            {
                                "index": i,
                                "name": name,
                                "value": value,
                                "weight": weight,
                                "value_density": value / weight,
                            }
                        )

                return {
                    "status": "optimal",
                    "capacity": capacity,
                    "selected_items": selected_list,
                    "total_value": total_value,
                    "total_weight": total_weight,
                    "weight_utilization": total_weight / capacity,
                    "objective_value": solution.objective_value,
                    "statistics": solution.statistics,
                }
            else:
                return {
                    "status": solution.status,
                    "error": "No feasible solution found",
                }
        case _:
            return {"status": "error", "error": "Failed to solve knapsack problem"}


def analyze_efficiency(results):
    """Analyze the efficiency of the knapsack solution."""
    if results["status"] != "optimal":
        return None

    selected_items = results["selected_items"]
    capacity = results["capacity"]

    # Calculate value density statistics
    densities = [item["value_density"] for item in selected_items]
    avg_density = sum(densities) / len(densities) if densities else 0

    # Calculate unused capacity
    unused_capacity = capacity - results["total_weight"]

    return {
        "average_value_density": avg_density,
        "unused_capacity": unused_capacity,
        "capacity_utilization": results["weight_utilization"],
        "items_selected": len(selected_items),
        "value_per_weight_unit": (
            results["total_value"] / results["total_weight"]
            if results["total_weight"] > 0
            else 0
        ),
    }


def print_results(results) -> None:
    """Print knapsack optimization results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Knapsack Problem Optimization Results")
    print("=" * 55)

    print(f"\nKnapsack Capacity: {results['capacity']} units")
    print(f"Total Value: {results['total_value']}")
    print(
        f"Total Weight: {results['total_weight']} / {results['capacity']} ({results['weight_utilization']:.1%})"
    )

    # Print selected items
    selected_items = results["selected_items"]
    print(f"\nSelected Items ({len(selected_items)} items):")
    print("-" * 55)
    print(f"{'Item':<15} {'Value':<8} {'Weight':<8} {'Density':<10}")
    print("-" * 55)

    for item in sorted(selected_items, key=lambda x: x["value_density"], reverse=True):
        print(
            f"{item['name']:<15} {item['value']:<8} {item['weight']:<8} {item['value_density']:<10.2f}"
        )

    # Efficiency analysis
    analysis = analyze_efficiency(results)
    if analysis:
        print("\nEfficiency Analysis:")
        print("-" * 25)
        print(f"Average value density: {analysis['average_value_density']:.2f}")
        print(f"Unused capacity: {analysis['unused_capacity']} units")
        print(f"Value per weight unit: {analysis['value_per_weight_unit']:.2f}")

    # Print solver statistics
    if "statistics" in results:
        print("\nSolver Statistics:")
        for key, value in results["statistics"].items():
            display_key = " ".join(word.title() for word in key.split("_"))
            print(f"  {display_key}: {value}")


def main() -> None:
    """Main function to run the knapsack optimization example."""
    print(__doc__)

    # Solve basic knapsack problem
    print("Solving basic knapsack problem...")
    results = solve_knapsack(capacity=100)
    print_results(results)


def test_knapsack() -> None:
    """Test function for pytest."""
    # Test basic knapsack
    results = solve_knapsack(capacity=100)
    assert results["status"] == "optimal"
    assert results["total_weight"] <= 100
    assert results["total_value"] > 0
    assert len(results["selected_items"]) > 0

    # Test that weight constraint is satisfied
    assert results["weight_utilization"] <= 1.0

    # Test efficiency analysis
    analysis = analyze_efficiency(results)
    assert analysis is not None
    assert analysis["average_value_density"] > 0
    assert analysis["capacity_utilization"] <= 1.0


if __name__ == "__main__":
    main()

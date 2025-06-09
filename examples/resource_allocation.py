"""
Resource Allocation Optimization Example

This module demonstrates resource allocation optimization for project selection.
The problem involves selecting which projects to fund while maximizing total value
subject to budget, resource, and strategic constraints.

The mixed-integer programming problem involves:
- Binary decision variables for project selection (fund or not)
- Resource constraints (budget, personnel, equipment)
- Strategic constraints (minimum projects per category)
- Risk diversification constraints
- Value maximization objective

This is a classic capital budgeting problem suitable for HiGHS mixed-integer programming.
"""

from returns.result import Failure, Success

from usolver_mcp.solvers.highs_solver import simple_highs_solver


def create_resource_allocation_problem():
    """
    Create a resource allocation optimization problem.

    The problem involves selecting projects from different categories:
    - Technology projects (high value, high risk)
    - Infrastructure projects (medium value, low risk)
    - Research projects (variable value, medium risk)
    - Marketing projects (medium value, low risk)

    Returns:
        dict: Problem parameters for the HiGHS solver
    """
    # Projects: (value, budget_cost, personnel_needed, equipment_needed, category, name)
    projects = [
        (100, 50, 5, 2, 0, "AI Platform"),  # Technology
        (150, 80, 8, 3, 0, "Cloud Migration"),  # Technology
        (80, 40, 3, 1, 0, "Mobile App"),  # Technology
        (120, 60, 6, 4, 1, "Data Center"),  # Infrastructure
        (90, 45, 4, 3, 1, "Network Upgrade"),  # Infrastructure
        (70, 35, 3, 2, 1, "Security System"),  # Infrastructure
        (110, 70, 10, 1, 2, "Drug Discovery"),  # Research
        (95, 55, 7, 1, 2, "Material Science"),  # Research
        (60, 30, 4, 1, 2, "Algorithm Research"),  # Research
        (85, 40, 6, 2, 3, "Brand Campaign"),  # Marketing
        (75, 35, 5, 1, 3, "Digital Marketing"),  # Marketing
        (55, 25, 3, 1, 3, "Market Research"),  # Marketing
    ]

    n_projects = len(projects)
    values = [p[0] for p in projects]
    budgets = [p[1] for p in projects]
    personnel = [p[2] for p in projects]
    equipment = [p[3] for p in projects]
    categories = [p[4] for p in projects]
    [p[5] for p in projects]

    # Available resources
    total_budget = 350  # Increased to make problem feasible
    total_personnel = 45  # Increased to make problem feasible
    total_equipment = 20  # Increased to make problem feasible

    # Variable definitions (binary: select project or not)
    variables = []
    for i in range(n_projects):
        variables.append(
            {"name": f"project_{i}", "lb": 0, "ub": 1, "type": "bin"}  # Binary variable
        )

    # Constraint matrix
    # Rows: [budget, personnel, equipment, tech_min, infra_min, research_min, marketing_min]
    constraint_matrix = []

    # Budget constraint
    constraint_matrix.append(budgets)

    # Personnel constraint
    constraint_matrix.append(personnel)

    # Equipment constraint
    constraint_matrix.append(equipment)

    # Minimum projects per category constraints
    # Technology projects (category 0): at least 1
    tech_row = [1 if categories[i] == 0 else 0 for i in range(n_projects)]
    constraint_matrix.append(tech_row)

    # Infrastructure projects (category 1): at least 1
    infra_row = [1 if categories[i] == 1 else 0 for i in range(n_projects)]
    constraint_matrix.append(infra_row)

    # Research projects (category 2): at least 1
    research_row = [1 if categories[i] == 2 else 0 for i in range(n_projects)]
    constraint_matrix.append(research_row)

    # Marketing projects (category 3): at least 1
    marketing_row = [1 if categories[i] == 3 else 0 for i in range(n_projects)]
    constraint_matrix.append(marketing_row)

    # Constraint senses and RHS values
    constraint_senses = ["<=", "<=", "<=", ">=", ">=", ">=", ">="]
    rhs_values = [
        total_budget,  # Budget limit
        total_personnel,  # Personnel limit
        total_equipment,  # Equipment limit
        1,  # Min 1 technology project
        1,  # Min 1 infrastructure project
        1,  # Min 1 research project
        1,  # Min 1 marketing project
    ]

    return {
        "sense": "maximize",
        "objective_coeffs": values,
        "variables": variables,
        "constraint_matrix": constraint_matrix,
        "constraint_senses": constraint_senses,
        "rhs_values": rhs_values,
        "description": "Resource allocation optimization for project portfolio selection",
        "projects": projects,
    }


def solve_resource_allocation():
    """
    Solve the resource allocation problem and return results.

    Returns:
        dict: Solution results including selected projects and resource usage
    """
    problem_params = create_resource_allocation_problem()

    # Extract solver parameters (exclude 'projects' which is for analysis)
    solver_params = {k: v for k, v in problem_params.items() if k != "projects"}
    result = simple_highs_solver(**solver_params)

    # Parse the result from the HiGHS solver
    match result:
        case Success(solution):
            if solution.status.value == "optimal":
                # Extract solution values
                solution_values = solution.solution or []

                # Map solution values to project selections
                selections = {}
                for i in range(len(problem_params["projects"])):
                    var_name = f"project_{i}"
                    if i < len(solution_values):
                        selections[var_name] = round(
                            solution_values[i]
                        )  # Binary variable
                    else:
                        selections[var_name] = 0

                return {
                    "status": "optimal",
                    "selections": selections,
                    "total_value": solution.objective_value or 0,
                }
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
    """Analyze the resource allocation solution."""
    if results["status"] != "optimal":
        return None

    projects = problem_params["projects"]
    selections = results["selections"]

    # Analyze selected projects
    selected_projects = []
    total_budget_used = 0
    total_personnel_used = 0
    total_equipment_used = 0
    category_counts = [0, 0, 0, 0]  # Tech, Infra, Research, Marketing

    for i, project in enumerate(projects):
        var_name = f"project_{i}"
        if selections.get(var_name, 0) == 1:
            value, budget, personnel, equipment, category, name = project
            selected_projects.append(
                {
                    "name": name,
                    "value": value,
                    "budget": budget,
                    "personnel": personnel,
                    "equipment": equipment,
                    "category": category,
                }
            )
            total_budget_used += budget
            total_personnel_used += personnel
            total_equipment_used += equipment
            category_counts[category] += 1

    return {
        "selected_projects": selected_projects,
        "resource_usage": {
            "budget": {"used": total_budget_used, "available": 350},
            "personnel": {"used": total_personnel_used, "available": 45},
            "equipment": {"used": total_equipment_used, "available": 20},
        },
        "category_distribution": {
            "Technology": category_counts[0],
            "Infrastructure": category_counts[1],
            "Research": category_counts[2],
            "Marketing": category_counts[3],
        },
    }


def print_results(results, problem_params) -> None:
    """Print resource allocation results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Resource Allocation Optimization Results")
    print("=" * 60)
    print(f"\nMaximum Portfolio Value: {results['total_value']}")

    # Analyze solution
    analysis = analyze_solution(results, problem_params)
    if not analysis:
        return

    selected_projects = analysis["selected_projects"]
    print(f"\nSelected Projects ({len(selected_projects)} projects):")
    print("-" * 60)
    print(
        f"{'Project':<20} {'Value':<8} {'Budget':<8} {'Personnel':<10} {'Equipment':<10}"
    )
    print("-" * 60)

    # Group by category
    categories = ["Technology", "Infrastructure", "Research", "Marketing"]
    for cat_idx, cat_name in enumerate(categories):
        cat_projects = [p for p in selected_projects if p["category"] == cat_idx]
        if cat_projects:
            print(f"\n{cat_name} Projects:")
            for project in cat_projects:
                print(
                    f"  {project['name']:<18} {project['value']:<8} {project['budget']:<8} "
                    f"{project['personnel']:<10} {project['equipment']:<10}"
                )

    # Resource utilization
    resource_usage = analysis["resource_usage"]
    print("\nResource Utilization:")
    print("-" * 30)
    for resource, data in resource_usage.items():
        used = data["used"]
        available = data["available"]
        utilization = used / available
        print(
            f"{resource.title():<12}: {used:3d} / {available:3d} ({utilization:6.1%})"
        )

    # Category distribution
    category_dist = analysis["category_distribution"]
    print("\nCategory Distribution:")
    print("-" * 25)
    for category, count in category_dist.items():
        print(f"{category:<15}: {count} project{'s' if count != 1 else ''}")


def main() -> None:
    """Main function to run the resource allocation example."""
    print(__doc__)

    problem_params = create_resource_allocation_problem()
    results = solve_resource_allocation()
    print_results(results, problem_params)


def test_resource_allocation() -> None:
    """Test function for pytest."""
    problem_params = create_resource_allocation_problem()
    results = solve_resource_allocation()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test resource constraints
    analysis = analyze_solution(results, problem_params)
    assert analysis is not None

    resource_usage = analysis["resource_usage"]
    assert resource_usage["budget"]["used"] <= 350
    assert resource_usage["personnel"]["used"] <= 45
    assert resource_usage["equipment"]["used"] <= 20

    # Test category constraints (at least 1 project per category)
    category_dist = analysis["category_distribution"]
    for _category, count in category_dist.items():
        assert count >= 1

    # Test that total value is positive
    assert results["total_value"] > 0

    # Test that selected projects are valid
    selected_projects = analysis["selected_projects"]
    assert len(selected_projects) > 0


if __name__ == "__main__":
    main()

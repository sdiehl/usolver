"""
Chained Solvers Optimization Example

This module demonstrates how to chain multiple optimization solvers to solve
a complex restaurant optimization problem. The problem is solved in two stages:
1. Table layout optimization using OR-Tools (combinatorial optimization)
2. Staff scheduling optimization using CVXPY (convex optimization)

The restaurant owner needs to:
- Optimize table layout to maximize seating capacity given space constraints
- Optimize staff scheduling based on the resulting capacity to minimize labor costs

This showcases how combinatorial and convex optimization can work together
to solve multi-stage business optimization problems.
"""

import numpy as np
from returns.result import Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYConstraint,
    CVXPYObjective,
    CVXPYProblem,
    CVXPYVariable,
)
from usolver_mcp.models.cvxpy_models import (
    ObjectiveType as CVXPYObjectiveType,
)
from usolver_mcp.models.ortools_models import (
    Constraint,
    Objective,
    ObjectiveType,
    Variable,
    VariableType,
)
from usolver_mcp.models.ortools_models import (
    Problem as ORToolsProblem,
)
from usolver_mcp.solvers.cvxpy_solver import solve_cvxpy_problem
from usolver_mcp.solvers.ortools_solver import solve_problem as solve_ortools_problem


def create_table_layout_problem():
    """
    Create the table layout optimization problem using OR-Tools.

    The problem determines how many of each table size to use to maximize
    seating capacity while respecting space and operational constraints.

    Returns:
        ORToolsProblem: The table layout optimization problem
    """
    variables = [
        Variable(
            name="tables_2_seater",
            type=VariableType.INTEGER,
            domain=(0, 15),
            description="Number of 2-seater tables",
        ),
        Variable(
            name="tables_4_seater",
            type=VariableType.INTEGER,
            domain=(0, 12),
            description="Number of 4-seater tables",
        ),
        Variable(
            name="tables_6_seater",
            type=VariableType.INTEGER,
            domain=(0, 8),
            description="Number of 6-seater tables",
        ),
    ]

    constraints = [
        # Space constraint (floor area in square meters)
        Constraint(
            expression="model.add(4*tables_2_seater + 6*tables_4_seater + 9*tables_6_seater <= 150)",
            description="Total floor space constraint (150 sq meters)",
        ),
        # Maximum number of tables (operational constraint)
        Constraint(
            expression="model.add(tables_2_seater + tables_4_seater + tables_6_seater <= 20)",
            description="Maximum number of tables constraint",
        ),
        # Ensure a minimum mix of table sizes for variety
        Constraint(
            expression="model.add(tables_2_seater >= 2)",
            description="Minimum number of 2-seater tables",
        ),
        Constraint(
            expression="model.add(tables_4_seater >= 3)",
            description="Minimum number of 4-seater tables",
        ),
        Constraint(
            expression="model.add(tables_6_seater >= 1)",
            description="Minimum number of 6-seater tables",
        ),
    ]

    # Objective: Maximize total seating capacity
    objective = Objective(
        type=ObjectiveType.MAXIMIZE,
        expression="2*tables_2_seater + 4*tables_4_seater + 6*tables_6_seater",
    )

    return ORToolsProblem(
        variables=variables,
        constraints=constraints,
        objective=objective,
        description="Restaurant table layout optimization to maximize seating capacity",
    )


def create_staff_scheduling_problem(total_seats):
    """
    Create the staff scheduling problem using CVXPY.

    Args:
        total_seats: Total seating capacity from the table layout solution

    Returns:
        CVXPYProblem: The staff scheduling optimization problem
    """
    # Operating hours (12-hour day)
    periods = list(range(12))
    n_periods = len(periods)

    # Define variables
    variables = [
        CVXPYVariable(
            name="staff_per_hour",
            shape=n_periods,
        )
    ]

    # Define constraints
    constraints = []

    # Staffing capacity: each staff member can handle up to 20 seats during peak
    for i in range(n_periods):
        constraints.append(
            CVXPYConstraint(
                expression=f"20 * staff_per_hour[{i}] >= {total_seats} * demand_factor[{i}]",
                description=f"Staff capacity constraint for hour {i}",
            )
        )

        # Minimum staff requirements (always need at least 2 staff)
        constraints.append(
            CVXPYConstraint(
                expression=f"staff_per_hour[{i}] >= 2",
                description=f"Minimum staff requirement for hour {i}",
            )
        )

    # Smooth staffing transitions (max 2 people change between hours)
    for i in range(n_periods - 1):
        constraints.append(
            CVXPYConstraint(
                expression=f"cp.abs(staff_per_hour[{i+1}] - staff_per_hour[{i}]) <= 2",
                description=f"Smooth staff transition between hours {i} and {i+1}",
            )
        )

    # Define objective: minimize total labor cost
    objective = CVXPYObjective(
        type=CVXPYObjectiveType.MINIMIZE,
        expression="cp.sum(cp.multiply(staff_per_hour, hourly_wage))",
    )

    # Demand patterns throughout the day (as fraction of capacity)
    demand_pattern = [
        0.4,
        0.4,
        0.6,
        0.8,
        1.0,
        1.0,  # Hours 0-5: gradual morning buildup
        0.8,
        0.6,
        0.8,
        1.0,
        0.8,
        0.4,  # Hours 6-11: lunch rush, dinner rush, wind down
    ]

    return CVXPYProblem(
        variables=variables,
        constraints=constraints,
        objective=objective,
        parameters={
            "demand_factor": demand_pattern,
            "hourly_wage": [25.0] * n_periods,  # $25/hour per staff member
        },
        description="Restaurant staff scheduling optimization to minimize labor costs",
    )


def solve_chained_optimization():
    """
    Solve the chained optimization problem and return results.

    Returns:
        dict: Combined results from both optimization stages
    """
    # Stage 1: Solve table layout problem
    table_problem = create_table_layout_problem()
    table_result = solve_ortools_problem(table_problem)

    match table_result:
        case Success(table_solution):
            if table_solution.is_feasible:
                # Extract table layout results
                solution_values = table_solution.values
                total_seats = (
                    2 * solution_values["tables_2_seater"]
                    + 4 * solution_values["tables_4_seater"]
                    + 6 * solution_values["tables_6_seater"]
                )

                table_results = {
                    "status": "optimal",
                    "tables_2_seater": solution_values["tables_2_seater"],
                    "tables_4_seater": solution_values["tables_4_seater"],
                    "tables_6_seater": solution_values["tables_6_seater"],
                    "total_seats": total_seats,
                    "objective_value": table_solution.objective_value,
                }

                # Stage 2: Solve staff scheduling problem using table layout results
                staff_problem = create_staff_scheduling_problem(total_seats)
                staff_result = solve_cvxpy_problem(staff_problem)

                match staff_result:
                    case Success(staff_solution):
                        if staff_solution.status == "optimal":
                            staff_results = {
                                "status": "optimal",
                                "staff_per_hour": staff_solution.values[
                                    "staff_per_hour"
                                ],
                                "total_daily_cost": staff_solution.objective_value,
                            }

                            return {
                                "overall_status": "optimal",
                                "table_layout": table_results,
                                "staff_scheduling": staff_results,
                            }
                        else:
                            return {
                                "overall_status": "staff_scheduling_failed",
                                "table_layout": table_results,
                                "error": f"Staff scheduling failed: {staff_solution.status}",
                            }
                    case _:
                        return {
                            "overall_status": "staff_scheduling_error",
                            "table_layout": table_results,
                            "error": "Error solving staff scheduling problem",
                        }
            else:
                return {
                    "overall_status": "table_layout_failed",
                    "error": f"Table layout optimization failed: {table_solution.status}",
                }
        case _:
            return {
                "overall_status": "table_layout_error",
                "error": "Error solving table layout problem",
            }


def print_results(results) -> None:
    """Print chained optimization results in a formatted way."""
    print("Chained Solvers Optimization Results")
    print("=" * 60)

    if results["overall_status"] != "optimal":
        print(f"Overall Status: {results['overall_status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    # Print table layout results
    table_results = results["table_layout"]
    print("\n1. Table Layout Optimization (OR-Tools):")
    print("-" * 45)
    print(f"2-seater tables: {table_results['tables_2_seater']}")
    print(f"4-seater tables: {table_results['tables_4_seater']}")
    print(f"6-seater tables: {table_results['tables_6_seater']}")
    print(f"Total seating capacity: {table_results['total_seats']} people")

    # Print staff scheduling results
    staff_results = results["staff_scheduling"]
    print("\n2. Staff Scheduling Optimization (CVXPY):")
    print("-" * 45)

    staff_per_hour = staff_results["staff_per_hour"]
    for hour, staff in enumerate(staff_per_hour):
        time_label = f"{hour}:00-{hour+1}:00"
        print(f"Hour {time_label}: {staff:.1f} staff members")

    print(f"\nTotal daily labor cost: ${staff_results['total_daily_cost']:.2f}")

    # Calculate some summary statistics
    avg_staff = np.mean(staff_per_hour)
    max_staff = np.max(staff_per_hour)
    min_staff = np.min(staff_per_hour)

    print("\nStaffing Summary:")
    print(f"Average staff per hour: {avg_staff:.1f}")
    print(f"Peak staffing: {max_staff:.1f}")
    print(f"Minimum staffing: {min_staff:.1f}")


def validate_solution(results) -> bool:
    """Validate that the chained solution is consistent and feasible."""
    if results["overall_status"] != "optimal":
        return False

    table_results = results["table_layout"]
    staff_results = results["staff_scheduling"]

    # Validate table layout
    total_tables = (
        table_results["tables_2_seater"]
        + table_results["tables_4_seater"]
        + table_results["tables_6_seater"]
    )

    if total_tables > 20:  # Max tables constraint
        return False

    # Validate minimum table requirements
    if (
        table_results["tables_2_seater"] < 2
        or table_results["tables_4_seater"] < 3
        or table_results["tables_6_seater"] < 1
    ):
        return False

    # Validate staff scheduling
    staff_per_hour = staff_results["staff_per_hour"]

    # Check minimum staffing (allow small numerical error)
    if any(staff < 1.99 for staff in staff_per_hour):
        return False

    # Check smooth transitions (max 2 people change)
    for i in range(len(staff_per_hour) - 1):
        if (
            abs(staff_per_hour[i + 1] - staff_per_hour[i]) > 2.01
        ):  # Allow small numerical error
            return False

    return True


def main() -> None:
    """Main function to run the chained solvers example."""
    print(__doc__)

    results = solve_chained_optimization()
    print_results(results)

    # Validate the solution
    if validate_solution(results):
        print("\n✓ Solution validation passed!")
    else:
        print("\n✗ Solution validation failed!")


def test_chained_solvers() -> None:
    """Test function for pytest."""
    results = solve_chained_optimization()

    # Test that we get an optimal solution
    assert results["overall_status"] == "optimal"

    # Test table layout results
    table_results = results["table_layout"]
    assert table_results["status"] == "optimal"
    assert table_results["total_seats"] > 0

    # Test minimum table requirements
    assert table_results["tables_2_seater"] >= 2
    assert table_results["tables_4_seater"] >= 3
    assert table_results["tables_6_seater"] >= 1

    # Test staff scheduling results
    staff_results = results["staff_scheduling"]
    assert staff_results["status"] == "optimal"
    assert staff_results["total_daily_cost"] > 0

    # Test staff constraints (allow small numerical error)
    staff_per_hour = staff_results["staff_per_hour"]
    assert all(staff >= 1.99 for staff in staff_per_hour)  # Minimum 2 staff

    # Test solution validation
    assert validate_solution(results)


if __name__ == "__main__":
    main()

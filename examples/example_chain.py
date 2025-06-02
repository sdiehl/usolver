"""
Restaurant Optimization Example - Chained Solver Demonstration

This example shows how to chain multiple solvers together to solve a complex
restaurant optimization problem:
1. First using OR-Tools to optimize table layout
2. Then using CVXPY to optimize staff scheduling based on the table layout results
"""

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
from usolver_mcp.solvers.ortools_solver import (
    solve_problem as solve_ortools_problem,
)


def create_table_layout_problem() -> ORToolsProblem:
    """
    Creates the table layout optimization problem using OR-Tools.

    The problem determines how many of each table size to use to maximize
    seating capacity while respecting space constraints.
    """
    variables = [
        Variable(
            name="tables_2_seater",
            type=VariableType.INTEGER,
            domain=(0, 10),
            description="Number of 2-seater tables",
        ),
        Variable(
            name="tables_4_seater",
            type=VariableType.INTEGER,
            domain=(0, 10),
            description="Number of 4-seater tables",
        ),
        Variable(
            name="tables_6_seater",
            type=VariableType.INTEGER,
            domain=(0, 8),
            description="Number of 6-seater tables",
        ),
    ]

    # Space constraints (assuming each table needs certain area in sq meters)
    constraints = [
        Constraint(
            expression="4*tables_2_seater + 6*tables_4_seater + 9*tables_6_seater <= 150",
            description="Total floor space constraint (150 sq meters)",
        ),
        Constraint(
            expression="tables_2_seater + tables_4_seater + tables_6_seater <= 20",
            description="Maximum number of tables constraint",
        ),
        # Ensure a mix of table sizes
        Constraint(
            expression="tables_2_seater >= 2",
            description="Minimum number of 2-seater tables",
        ),
        Constraint(
            expression="tables_4_seater >= 3",
            description="Minimum number of 4-seater tables",
        ),
        Constraint(
            expression="tables_6_seater >= 1",
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
        description="Restaurant table layout optimization",
    )


def create_staff_scheduling_problem(total_seats: int) -> CVXPYProblem:
    """
    Creates the staff scheduling problem using CVXPY.

    Args:
        total_seats: Total seating capacity from the table layout solution

    Returns:
        CVXPYProblem instance for staff scheduling optimization
    """
    # Time periods (hours from opening)
    periods = list(range(12))  # 12-hour operating day

    # Define variables
    variables = [
        CVXPYVariable(
            name="staff_per_hour",
            shape=len(periods),
        )
    ]

    # Define constraints
    constraints = []

    # Each staff member can handle up to 20 seats during peak hours
    for i in range(len(periods)):
        constraints.append(
            CVXPYConstraint(
                expression=f"20 * staff_per_hour[{i}] >= {total_seats} * peak_factor[{i}]",
                description=f"Staff capacity constraint for hour {i}",
            )
        )
        # Minimum staff requirements
        constraints.append(
            CVXPYConstraint(
                expression=f"staff_per_hour[{i}] >= 2",
                description=f"Minimum staff requirement for hour {i}",
            )
        )

    # Staff change between consecutive hours should be smooth
    for i in range(len(periods) - 1):
        constraints.append(
            CVXPYConstraint(
                expression=f"cp.abs(staff_per_hour[{i+1}] - staff_per_hour[{i}]) <= 2",
                description=f"Smooth staff transition between hours {i} and {i+1}",
            )
        )

    # Define objective
    objective = CVXPYObjective(
        type=CVXPYObjectiveType.MINIMIZE,
        expression="cp.sum(cp.multiply(staff_per_hour, hourly_wage))",
    )

    return CVXPYProblem(
        variables=variables,
        constraints=constraints,
        objective=objective,
        parameters={
            "peak_factor": [
                0.4,
                0.4,
                0.6,
                0.8,
                1.0,
                1.0,
                0.8,
                0.6,
                0.8,
                1.0,
                0.8,
                0.4,
            ],
            "hourly_wage": [25.0] * len(periods),
        },
        description="Restaurant staff scheduling optimization",
    )


def main() -> None:
    # Step 1: Solve the table layout problem
    print("\n1. Solving table layout optimization...")
    table_problem = create_table_layout_problem()
    table_result = solve_ortools_problem(table_problem)

    match table_result:
        case Success(solution):
            if solution.is_feasible:
                print("\nTable Layout Solution:")
                print(f"Status: {solution.status}")

                # Extract results
                solution_values = solution.values
                total_seats = (
                    2 * solution_values["tables_2_seater"]
                    + 4 * solution_values["tables_4_seater"]
                    + 6 * solution_values["tables_6_seater"]
                )

                print(f"2-seater tables: {solution_values['tables_2_seater']}")
                print(f"4-seater tables: {solution_values['tables_4_seater']}")
                print(f"6-seater tables: {solution_values['tables_6_seater']}")
                print(f"Total seating capacity: {total_seats}")

                # Step 2: Use the results to solve the staff scheduling problem
                print("\n2. Solving staff scheduling optimization...")
                staff_problem = create_staff_scheduling_problem(total_seats)
                staff_result = solve_cvxpy_problem(staff_problem)

                match staff_result:
                    case Success(staff_solution):
                        if staff_solution.status == "optimal":
                            print("\nStaff Scheduling Solution:")
                            print(f"Status: {staff_solution.status}")

                            staff_per_hour = staff_solution.values["staff_per_hour"]
                            for hour, staff in enumerate(staff_per_hour):
                                print(f"Hour {hour}: {staff:.1f} staff members needed")

                            print(
                                f"\nTotal daily labor cost: ${staff_solution.objective_value:.2f}"
                            )
                        else:
                            print("Could not find a valid staff schedule.")
                            print(f"Status: {staff_solution.status}")
                    case _:
                        print("Error solving the staff scheduling problem")
            else:
                print("Could not find a valid table layout.")
                print(f"Status: {solution.status}")
        case _:
            print("Error solving the table layout problem")


if __name__ == "__main__":
    main()

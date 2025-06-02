"""Example of a nurse scheduling problem using the usolver OR-Tools interface."""

from returns.result import Success

from usolver_mcp.models.ortools_models import (
    Constraint,
    Problem,
    Variable,
    VariableType,
)
from usolver_mcp.solvers.ortools_solver import solve_problem


def main() -> None:
    # Data
    nurses = ["Alice", "Bob", "Charlie", "Diana"]  # Our nursing team
    shifts = ["Morning (7AM-3PM)", "Evening (3PM-11PM)", "Night (11PM-7AM)"]
    days = ["Monday", "Tuesday", "Wednesday"]

    num_nurses = len(nurses)
    num_shifts = len(shifts)
    num_days = len(days)

    # Create variables
    # shifts_var[n,d,s]: nurse 'n' works shift 's' on day 'd'
    shifts_var = Variable(
        name="shifts_var",  # This is the key in the variables dictionary
        type=VariableType.BOOLEAN,
        shape=[num_nurses, num_days, num_shifts],
        description="Binary variable indicating if a nurse works a particular shift on a particular day",
    )

    # Create constraints
    constraints = []

    # Each shift must be assigned to exactly one nurse each day
    for d in range(num_days):
        for s in range(num_shifts):
            constraints.append(
                Constraint(
                    expression=f"model.add(sum([shifts_var[n][{d}][{s}] for n in range({num_nurses})]) == 1)",
                    description=f"Exactly one nurse must work the {shifts[s]} shift on {days[d]}",
                )
            )

    # Each nurse works at most one shift per day
    for n in range(num_nurses):
        for d in range(num_days):
            constraints.append(
                Constraint(
                    expression=f"model.add(sum([shifts_var[{n}][{d}][s] for s in range({num_shifts})]) <= 1)",
                    description=f"{nurses[n]} works at most one shift on {days[d]}",
                )
            )

    # Try to distribute the shifts evenly
    min_shifts_per_nurse = (num_shifts * num_days) // num_nurses
    max_shifts_per_nurse = min_shifts_per_nurse + (
        1 if num_shifts * num_days % num_nurses else 0
    )

    for n in range(num_nurses):
        # Each nurse must work at least min_shifts_per_nurse shifts
        constraints.append(
            Constraint(
                expression=(
                    f"model.add(sum([shifts_var[{n}][d][s] "
                    f"for d in range({num_days}) "
                    f"for s in range({num_shifts})]) >= {min_shifts_per_nurse})"
                ),
                description=f"{nurses[n]} must work at least {min_shifts_per_nurse} shifts",
            )
        )
        # Each nurse must work at most max_shifts_per_nurse shifts
        constraints.append(
            Constraint(
                expression=(
                    f"model.add(sum([shifts_var[{n}][d][s] "
                    f"for d in range({num_days}) "
                    f"for s in range({num_shifts})]) <= {max_shifts_per_nurse})"
                ),
                description=f"{nurses[n]} must work at most {max_shifts_per_nurse} shifts",
            )
        )

    # Create and solve the problem
    problem = Problem(
        variables=[shifts_var],
        constraints=constraints,
        description="Hospital nurse scheduling problem",
        parameters={"enumerate_all_solutions": True},
    )

    result = solve_problem(problem)

    match result:
        case Success(solution):
            if solution.is_feasible:
                print("Found a valid schedule!")
                print(f"Status: {solution.status}")

                # Extract the solution values
                schedule_val: list[list[list[bool]]] = solution.values["shifts_var"]

                # Print the schedule in a more readable format
                print("\nWeekly Schedule:")
                print("=" * 60)

                for d in range(num_days):
                    print(f"\n{days[d]}:")
                    print("-" * 20)

                    # Print each shift
                    for s in range(num_shifts):
                        # Find which nurse is working this shift
                        for n in range(num_nurses):
                            if schedule_val[n][d][s]:
                                print(f"{shifts[s]}: {nurses[n]}")
                                break

                print("\nSchedule Statistics:")
                print("=" * 60)

                # Print shifts per nurse
                for n in range(num_nurses):
                    nurse_shifts = sum(
                        schedule_val[n][d][s]
                        for d in range(num_days)
                        for s in range(num_shifts)
                    )
                    print(f"{nurses[n]}: {nurse_shifts} shifts")

                # Print solver statistics
                print("\nSolver Statistics:")
                print("=" * 60)
                for key, value in solution.statistics.items():
                    # Convert snake_case to Title Case for display
                    display_key = " ".join(word.title() for word in key.split("_"))
                    print(f"{display_key}: {value}")
            else:
                print("Could not find a valid schedule.")
                print(f"Status: {solution.status}")
        case _:
            print("Error solving the scheduling problem")


if __name__ == "__main__":
    main()

"""
Nurse Scheduling Optimization Example

This module demonstrates staff scheduling optimization for a hospital nursing department.
The problem involves assigning nurses to shifts across multiple days while satisfying
operational requirements and work-life balance constraints.

The constraint programming problem assigns nurses to shifts subject to:
- Each shift must be assigned to exactly one nurse each day
- Each nurse works at most one shift per day
- Workload distribution: each nurse works 2-3 shifts over the period
- Special constraints (e.g., certain nurses unavailable on specific days)
- Minimum staffing requirements per shift type

This is a classic constraint satisfaction problem suitable for OR-Tools CP-SAT.
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


def create_nurse_scheduling_problem():
    """
    Create a nurse scheduling optimization problem.

    Returns:
        Problem: The OR-Tools constraint programming problem
    """
    # Data
    nurses = ["Alice", "Bob", "Charlie", "Diana"]
    shifts = ["Morning (7AM-3PM)", "Evening (3PM-11PM)", "Night (11PM-7AM)"]
    days = ["Monday", "Tuesday", "Wednesday"]

    num_nurses = len(nurses)
    num_shifts = len(shifts)
    num_days = len(days)

    # Create decision variables
    # shifts_var[n,d,s]: nurse 'n' works shift 's' on day 'd'
    shifts_var = Variable(
        name="shifts_var",
        type=VariableType.BOOLEAN,
        shape=[num_nurses, num_days, num_shifts],
        description="Binary variable indicating if a nurse works a particular shift on a particular day",
    )

    # Create constraints
    constraints = []

    # Constraint 1: Each shift must be assigned to exactly one nurse each day
    for d in range(num_days):
        for s in range(num_shifts):
            constraints.append(
                Constraint(
                    expression=f"model.add(sum([shifts_var[n][{d}][{s}] for n in range({num_nurses})]) == 1)",
                    description=f"Exactly one nurse must work the {shifts[s]} shift on {days[d]}",
                )
            )

    # Constraint 2: Each nurse works at most one shift per day
    for n in range(num_nurses):
        for d in range(num_days):
            constraints.append(
                Constraint(
                    expression=f"model.add(sum([shifts_var[{n}][{d}][s] for s in range({num_shifts})]) <= 1)",
                    description=f"{nurses[n]} works at most one shift on {days[d]}",
                )
            )

    # Constraint 3: Distribute shifts evenly (2-3 shifts per nurse)
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

    # Constraint 4: Charlie can't work on Tuesday (example of special constraint)
    charlie_index = nurses.index("Charlie")
    tuesday_index = days.index("Tuesday")
    for s in range(num_shifts):
        constraints.append(
            Constraint(
                expression=f"model.add(shifts_var[{charlie_index}][{tuesday_index}][{s}] == 0)",
                description="Charlie cannot work any shift on Tuesday",
            )
        )

    # Optional: Add an objective to minimize schedule disruption
    # (e.g., minimize night shifts followed by morning shifts)
    objective = Objective(
        type=ObjectiveType.FEASIBILITY, description="Find any feasible schedule"
    )

    return Problem(
        variables=[shifts_var],
        constraints=constraints,
        objective=objective,
        description="Hospital nurse scheduling problem with fairness and availability constraints",
        parameters={"enumerate_all_solutions": False},
    )


def solve_nurse_scheduling():
    """
    Solve the nurse scheduling problem and return results.

    Returns:
        dict: Solution results including schedule and statistics
    """
    problem = create_nurse_scheduling_problem()
    result = solve_problem(problem)

    nurses = ["Alice", "Bob", "Charlie", "Diana"]
    shifts = ["Morning (7AM-3PM)", "Evening (3PM-11PM)", "Night (11PM-7AM)"]
    days = ["Monday", "Tuesday", "Wednesday"]

    match result:
        case Success(solution):
            if solution.is_feasible:
                schedule_val = solution.values["shifts_var"]

                # Create readable schedule
                schedule = {}
                for d, day in enumerate(days):
                    schedule[day] = {}
                    for s, shift in enumerate(shifts):
                        for n, nurse in enumerate(nurses):
                            if schedule_val[n][d][s]:
                                schedule[day][shift] = nurse
                                break

                # Calculate nurse workloads
                nurse_shifts = {}
                for n, nurse in enumerate(nurses):
                    total_shifts = sum(
                        schedule_val[n][d][s]
                        for d in range(len(days))
                        for s in range(len(shifts))
                    )
                    nurse_shifts[nurse] = total_shifts

                return {
                    "status": "feasible",
                    "schedule": schedule,
                    "nurse_shifts": nurse_shifts,
                    "statistics": solution.statistics,
                }
            else:
                return {
                    "status": solution.status,
                    "error": "No feasible schedule found",
                }
        case _:
            return {"status": "error", "error": "Failed to solve problem"}


def print_results(results) -> None:
    """Print nurse scheduling results in a formatted way."""
    if results["status"] != "feasible":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Nurse Scheduling Optimization Results")
    print("=" * 60)

    # Print weekly schedule
    schedule = results["schedule"]
    print("\nWeekly Schedule:")
    print("-" * 60)

    for day, shifts_for_day in schedule.items():
        print(f"\n{day}:")
        for shift, nurse in shifts_for_day.items():
            print(f"  {shift}: {nurse}")

    # Print workload distribution
    nurse_shifts = results["nurse_shifts"]
    print("\nWorkload Distribution:")
    print("-" * 30)
    for nurse, shifts in nurse_shifts.items():
        print(f"{nurse}: {shifts} shifts")

    # Print solver statistics
    if "statistics" in results:
        print("\nSolver Statistics:")
        print("-" * 30)
        for key, value in results["statistics"].items():
            display_key = " ".join(word.title() for word in key.split("_"))
            print(f"{display_key}: {value}")


def validate_schedule(results) -> bool:
    """Validate that the schedule meets all constraints."""
    if results["status"] != "feasible":
        return False

    schedule = results["schedule"]

    # Check that each shift is assigned
    for _day, shifts_for_day in schedule.items():
        if len(shifts_for_day) != 3:  # Should have 3 shifts per day
            return False

    # Check Charlie doesn't work on Tuesday
    if "Charlie" in schedule.get("Tuesday", {}).values():
        return False

    # Check workload distribution (2-3 shifts per nurse)
    nurse_shifts = results["nurse_shifts"]
    for _nurse, shifts in nurse_shifts.items():
        if not (2 <= shifts <= 3):
            return False

    return True


def main() -> None:
    """Main function to run the nurse scheduling example."""
    print(__doc__)

    results = solve_nurse_scheduling()
    print_results(results)

    # Validate the solution
    if validate_schedule(results):
        print("\n✓ Schedule validation passed!")
    else:
        print("\n✗ Schedule validation failed!")


def test_nurse_scheduling() -> None:
    """Test function for pytest."""
    results = solve_nurse_scheduling()

    # Test that we get a feasible solution
    assert results["status"] == "feasible"

    # Test that we have a complete schedule
    schedule = results["schedule"]
    assert len(schedule) == 3  # 3 days

    for day_schedule in schedule.values():
        assert len(day_schedule) == 3  # 3 shifts per day

    # Test workload distribution
    nurse_shifts = results["nurse_shifts"]
    for _nurse, shifts in nurse_shifts.items():
        assert 2 <= shifts <= 3  # Each nurse works 2-3 shifts

    # Test special constraint: Charlie doesn't work on Tuesday
    assert "Charlie" not in schedule.get("Tuesday", {}).values()

    # Test that schedule is valid
    assert validate_schedule(results)


if __name__ == "__main__":
    main()

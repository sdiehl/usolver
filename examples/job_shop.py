"""
Job Shop Scheduling Optimization

This module demonstrates how to solve job shop scheduling problems using constraint programming.
Job shop scheduling involves scheduling a set of jobs on a set of machines, where each job
consists of a sequence of operations that must be performed in a specific order on specific machines.

The goal is to minimize the total completion time (makespan) while respecting:
- Operation precedence within each job
- Machine capacity constraints (one operation per machine at a time)
- Non-preemptive processing (operations cannot be interrupted)

This implementation uses OR-Tools constraint programming solver to handle the complex
scheduling constraints and optimization objectives.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def solve_job_shop_problem(
    jobs_data: list[list[tuple[int, int]]], num_machines: int
) -> dict | None:
    """
    Solve a job shop scheduling problem.

    Args:
        jobs_data: List where jobs_data[j] is a list of (machine, duration) tuples for job j
        num_machines: Total number of machines available

    Returns:
        Dictionary containing schedule information if solution found, None otherwise
    """
    try:
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

        num_jobs = len(jobs_data)

        # Calculate bounds
        all_durations = []
        for job in jobs_data:
            for _, duration in job:
                all_durations.append(duration)

        max(all_durations) if all_durations else 100
        horizon = sum(all_durations)  # Upper bound on makespan

        logger.info(f"Job shop problem: {num_jobs} jobs, {num_machines} machines")
        logger.info(f"Time horizon: {horizon}")

        # Define variables
        variables = []
        constraints = []

        # For each operation, we need start time and end time
        # Variable naming: start_j_o for start time of operation o in job j
        #                 end_j_o for end time of operation o in job j
        for job_id, job in enumerate(jobs_data):
            for op_id, (_machine, duration) in enumerate(job):
                # Start time variable
                variables.append(
                    Variable(
                        name=f"start_{job_id}_{op_id}",
                        type=VariableType.INTEGER,
                        domain=(0, horizon - duration),
                        description=f"Start time of operation {op_id} in job {job_id}",
                    )
                )

                # End time variable
                variables.append(
                    Variable(
                        name=f"end_{job_id}_{op_id}",
                        type=VariableType.INTEGER,
                        domain=(duration, horizon),
                        description=f"End time of operation {op_id} in job {job_id}",
                    )
                )

                # Link start and end times
                constraints.append(
                    Constraint(
                        expression=f"model.add(end_{job_id}_{op_id} == start_{job_id}_{op_id} + {duration})",
                        description=f"Duration constraint for job {job_id}, operation {op_id}",
                    )
                )

        # Precedence constraints within each job
        for job_id, job in enumerate(jobs_data):
            for op_id in range(len(job) - 1):
                constraints.append(
                    Constraint(
                        expression=f"model.add(start_{job_id}_{op_id + 1} >= end_{job_id}_{op_id})",
                        description=f"Precedence in job {job_id}: op {op_id} before op {op_id + 1}",
                    )
                )

        # Machine capacity constraints (no two operations on same machine overlap)
        machine_operations: list[list[tuple[int, int]]] = [
            [] for _ in range(num_machines)
        ]
        for job_id, job in enumerate(jobs_data):
            for op_id, (machine, _duration) in enumerate(job):
                machine_operations[machine].append((job_id, op_id))

        for _machine_id, operations in enumerate(machine_operations):
            if len(operations) <= 1:
                continue

            for i, (job1, op1) in enumerate(operations):
                for job2, op2 in operations[i + 1 :]:
                    # Either operation 1 finishes before operation 2 starts, or vice versa
                    # We'll use auxiliary boolean variables for this disjunctive constraint
                    bool_var_name = f"before_{job1}_{op1}_{job2}_{op2}"
                    variables.append(
                        Variable(
                            name=bool_var_name,
                            type=VariableType.BOOLEAN,
                            description=f"True if job {job1} op {op1} before job {job2} op {op2}",
                        )
                    )

                    # Use a large constant M for big-M constraints
                    big_m = horizon

                    # If bool_var is true, then job1_op1 ends before job2_op2 starts
                    constraints.append(
                        Constraint(
                            expression=(
                                f"model.add(end_{job1}_{op1} <= start_{job2}_{op2} + "
                                f"{big_m} * (1 - {bool_var_name}))"
                            ),
                            description=f"If {bool_var_name}, then job {job1} op {op1} before job {job2} op {op2}",
                        )
                    )

                    # If bool_var is false, then job2_op2 ends before job1_op1 starts
                    constraints.append(
                        Constraint(
                            expression=f"model.add(end_{job2}_{op2} <= start_{job1}_{op1} + {big_m} * {bool_var_name})",
                            description=f"If not {bool_var_name}, then job {job2} op {op2} before job {job1} op {op1}",
                        )
                    )

        # Makespan variable (completion time of all jobs)
        variables.append(
            Variable(
                name="makespan",
                type=VariableType.INTEGER,
                domain=(0, horizon),
                description="Total completion time (makespan)",
            )
        )

        # Makespan is at least the end time of all operations
        for job_id, job in enumerate(jobs_data):
            last_op = len(job) - 1
            constraints.append(
                Constraint(
                    expression=f"model.add(makespan >= end_{job_id}_{last_op})",
                    description=f"Makespan at least completion time of job {job_id}",
                )
            )

        # Create objective
        objective = Objective(type=ObjectiveType.MINIMIZE, expression="makespan")

        # Create problem
        problem = Problem(
            variables=variables,
            constraints=constraints,
            objective=objective,
            description=f"Job shop scheduling: {num_jobs} jobs on {num_machines} machines",
        )

        logger.info("Solving job shop scheduling problem...")
        result = solve_problem(problem)

        if isinstance(result, Success):
            solution = result.unwrap()
            if solution.is_feasible:
                return parse_job_shop_solution_from_values(
                    solution.values, jobs_data, num_machines
                )

        logger.warning("No solution found for job shop problem")
        return None

    except Exception as e:
        logger.error(f"Error solving job shop problem: {e}")
        return None


def parse_job_shop_solution_from_values(
    values: dict[str, int], jobs_data: list[list[tuple[int, int]]], num_machines: int
) -> dict:
    """Parse the OR-Tools solution values for job shop scheduling."""
    solution = {
        "status": "SOLVED",
        "jobs": [],
        "machines": [[] for _ in range(num_machines)],
        "makespan": values.get("makespan", 0),
    }

    # Build job schedules
    for job_id, job in enumerate(jobs_data):
        job_schedule = []
        for op_id, (machine, duration) in enumerate(job):
            start_time = values.get(f"start_{job_id}_{op_id}", 0)
            end_time = values.get(f"end_{job_id}_{op_id}", duration)

            operation = {
                "operation": op_id,
                "machine": machine,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
            }
            job_schedule.append(operation)

            # Add to machine schedule
            if machine < len(solution["machines"]):
                solution["machines"][machine].append(
                    {
                        "job": job_id,
                        "operation": op_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                )

        solution["jobs"].append(job_schedule)

    # Sort machine schedules by start time
    for machine_schedule in solution["machines"]:
        machine_schedule.sort(key=lambda x: x["start_time"])

    return solution


def parse_job_shop_solution(
    result_text: str, jobs_data: list[list[tuple[int, int]]], num_machines: int
) -> dict:
    """Parse the OR-Tools solution for job shop scheduling."""
    solution = {
        "status": "SOLVED",
        "jobs": [],
        "machines": [[] for _ in range(num_machines)],
        "makespan": 0,
    }

    # Extract variable values from result
    variables = {}
    lines = result_text.split("\n")
    for line in lines:
        if "=" in line and ("start_" in line or "end_" in line or "makespan" in line):
            parts = line.strip().split("=")
            if len(parts) == 2:
                var_name = parts[0].strip()
                try:
                    value = int(parts[1].strip())
                    variables[var_name] = value
                except ValueError:
                    continue

    # Extract makespan
    solution["makespan"] = variables.get("makespan", 0)

    # Build job schedules
    for job_id, job in enumerate(jobs_data):
        job_schedule = []
        for op_id, (machine, duration) in enumerate(job):
            start_time = variables.get(f"start_{job_id}_{op_id}", 0)
            end_time = variables.get(f"end_{job_id}_{op_id}", duration)

            operation = {
                "operation": op_id,
                "machine": machine,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
            }
            job_schedule.append(operation)

            # Add to machine schedule
            if machine < len(solution["machines"]):
                solution["machines"][machine].append(
                    {
                        "job": job_id,
                        "operation": op_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                )

        solution["jobs"].append(job_schedule)

    # Sort machine schedules by start time
    for machine_schedule in solution["machines"]:
        machine_schedule.sort(key=lambda x: x["start_time"])

    return solution


def create_sample_job_shop() -> tuple[list[list[tuple[int, int]]], int]:
    """Create a sample job shop problem instance."""
    # Classic 3x3 job shop problem
    # 3 jobs, 3 machines
    # Format: (machine_id, duration)
    jobs_data = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0: M0(3) -> M1(2) -> M2(2)
        [(0, 2), (2, 1), (1, 4)],  # Job 1: M0(2) -> M2(1) -> M1(4)
        [(1, 4), (2, 3)],  # Job 2: M1(4) -> M2(3)
    ]
    num_machines = 3

    return jobs_data, num_machines


def create_large_job_shop() -> tuple[list[list[tuple[int, int]]], int]:
    """Create a larger job shop problem instance."""
    # 4 jobs, 4 machines
    jobs_data = [
        [(0, 5), (1, 3), (2, 4), (3, 2)],  # Job 0
        [(1, 2), (0, 6), (3, 3), (2, 1)],  # Job 1
        [(2, 3), (3, 5), (0, 2), (1, 4)],  # Job 2
        [(3, 4), (2, 2), (1, 3), (0, 5)],  # Job 3
    ]
    num_machines = 4

    return jobs_data, num_machines


def validate_job_shop_solution(
    solution: dict, jobs_data: list[list[tuple[int, int]]]
) -> bool:
    """Validate that a job shop solution satisfies all constraints."""
    try:
        # Check precedence constraints within each job
        for job_id, job_schedule in enumerate(solution["jobs"]):
            for i in range(len(job_schedule) - 1):
                if job_schedule[i]["end_time"] > job_schedule[i + 1]["start_time"]:
                    logger.error(f"Precedence violation in job {job_id}")
                    return False

        # Check machine capacity constraints
        for machine_id, machine_schedule in enumerate(solution["machines"]):
            for i in range(len(machine_schedule) - 1):
                if (
                    machine_schedule[i]["end_time"]
                    > machine_schedule[i + 1]["start_time"]
                ):
                    logger.error(f"Machine capacity violation on machine {machine_id}")
                    return False

        # Check duration constraints
        for job_id, job_schedule in enumerate(solution["jobs"]):
            for op in job_schedule:
                expected_duration = jobs_data[job_id][op["operation"]][1]
                actual_duration = op["end_time"] - op["start_time"]
                if actual_duration != expected_duration:
                    logger.error(
                        f"Duration violation in job {job_id}, operation {op['operation']}"
                    )
                    return False

        return True

    except Exception as e:
        logger.error(f"Error validating solution: {e}")
        return False


def print_job_shop_analysis(
    solution: dict, jobs_data: list[list[tuple[int, int]]], num_machines: int
) -> None:
    """Print detailed analysis of the job shop scheduling solution."""
    print("\n" + "=" * 70)
    print("JOB SHOP SCHEDULING SOLUTION")
    print("=" * 70)

    print("\nProblem Size:")
    print(f"  Jobs: {len(jobs_data)}")
    print(f"  Machines: {num_machines}")
    print(f"  Total Operations: {sum(len(job) for job in jobs_data)}")

    print(f"\nOptimal Makespan: {solution['makespan']}")

    print("\nJob Schedules:")
    for job_id, job_schedule in enumerate(solution["jobs"]):
        print(f"\n  Job {job_id}:")
        for op in job_schedule:
            print(
                f"    Op {op['operation']}: Machine {op['machine']} "
                f"[{op['start_time']}-{op['end_time']}] (duration: {op['duration']})"
            )

    print("\nMachine Schedules:")
    for machine_id, machine_schedule in enumerate(solution["machines"]):
        print(f"\n  Machine {machine_id}:")
        if not machine_schedule:
            print("    No operations assigned")
        else:
            for task in machine_schedule:
                print(
                    f"    Job {task['job']}.{task['operation']}: "
                    f"[{task['start_time']}-{task['end_time']}] (duration: {task['duration']})"
                )

    # Calculate machine utilization
    print("\nMachine Utilization:")
    for machine_id, machine_schedule in enumerate(solution["machines"]):
        total_work_time = sum(task["duration"] for task in machine_schedule)
        utilization = (
            (total_work_time / solution["makespan"]) * 100
            if solution["makespan"] > 0
            else 0
        )
        print(
            f"  Machine {machine_id}: {total_work_time}/{solution['makespan']} = {utilization:.1f}%"
        )

    # Validation
    is_valid = validate_job_shop_solution(solution, jobs_data)
    print(f"\nSolution Validation: {'✓ VALID' if is_valid else '✗ INVALID'}")


def main() -> None:
    """Main function to demonstrate job shop scheduling."""
    print("USolver Job Shop Scheduling Optimizer")
    print("====================================")

    # Solve small problem
    print("\n1. Small Job Shop Problem (3 jobs, 3 machines)")
    jobs_data, num_machines = create_sample_job_shop()

    print("\nProblem Definition:")
    for job_id, job in enumerate(jobs_data):
        operations = " -> ".join(
            [f"M{machine}({duration})" for machine, duration in job]
        )
        print(f"  Job {job_id}: {operations}")

    solution = solve_job_shop_problem(jobs_data, num_machines)

    if solution:
        print_job_shop_analysis(solution, jobs_data, num_machines)
    else:
        print("No solution found for small problem")

    # Solve larger problem
    print("\n" + "=" * 70)
    print("\n2. Larger Job Shop Problem (4 jobs, 4 machines)")
    jobs_data2, num_machines2 = create_large_job_shop()

    print("\nProblem Definition:")
    for job_id, job in enumerate(jobs_data2):
        operations = " -> ".join(
            [f"M{machine}({duration})" for machine, duration in job]
        )
        print(f"  Job {job_id}: {operations}")

    solution2 = solve_job_shop_problem(jobs_data2, num_machines2)

    if solution2:
        print_job_shop_analysis(solution2, jobs_data2, num_machines2)
    else:
        print("No solution found for larger problem")


def test_job_shop_scheduler() -> None:
    """Test function for pytest compatibility."""
    # Test small problem
    jobs_data, num_machines = create_sample_job_shop()
    solution = solve_job_shop_problem(jobs_data, num_machines)

    assert solution is not None, "Should find solution for small job shop problem"
    assert solution["makespan"] > 0, "Makespan should be positive"
    assert validate_job_shop_solution(solution, jobs_data), "Solution should be valid"
    assert len(solution["jobs"]) == len(jobs_data), "Should have schedule for each job"
    assert (
        len(solution["machines"]) == num_machines
    ), "Should have schedule for each machine"

    # Test that makespan is reasonable (should be at least the longest job duration)
    min_makespan = max(sum(duration for _, duration in job) for job in jobs_data)
    assert (
        solution["makespan"] >= min_makespan
    ), f"Makespan {solution['makespan']} should be at least {min_makespan}"

    print("All job shop scheduling tests passed!")


if __name__ == "__main__":
    main()

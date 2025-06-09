"""
N-Queens Problem Example

This module solves the classic N-Queens problem using constraint programming.
The problem involves placing N queens on an NxN chessboard such that no two
queens can attack each other (no two queens share the same row, column, or diagonal).

The constraint programming problem involves:
- N queen positions on an NxN board
- Each row must contain exactly one queen
- Each column must contain exactly one queen
- No two queens can be on the same diagonal
- Optional: find all solutions or just one solution

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


def create_nqueens_problem(n=8, find_all_solutions=False):
    """
    Create an N-Queens constraint programming problem.

    Args:
        n: Size of the chessboard (nxn) and number of queens
        find_all_solutions: Whether to enumerate all solutions

    Returns:
        Problem: The OR-Tools constraint programming problem
    """
    # Variables: queens[i] represents the column position of the queen in row i
    # Domain: each queen can be in any column from 0 to n-1
    queens = Variable(
        name="queens",
        type=VariableType.INTEGER,
        domain=(0, n - 1),
        shape=[n],
        description=f"Column position of queen in each row (0 to {n-1})",
    )

    constraints = []

    # Constraint 1: All queens must be in different columns
    # This is automatically handled by the AllDifferent constraint
    constraints.append(
        Constraint(
            expression=f"model.AddAllDifferent([queens[i] for i in range({n})])",
            description="All queens must be in different columns",
        )
    )

    # Constraint 2: No two queens on the same diagonal
    # For diagonal constraints, we need:
    # - No two queens on the same positive diagonal: queens[i] + i ≠ queens[j] + j for i ≠ j
    # - No two queens on the same negative diagonal: queens[i] - i ≠ queens[j] - j for i ≠ j

    # Positive diagonal constraint
    constraints.append(
        Constraint(
            expression=f"model.AddAllDifferent([queens[i] + i for i in range({n})])",
            description="No two queens on the same positive diagonal",
        )
    )

    # Negative diagonal constraint
    constraints.append(
        Constraint(
            expression=f"model.AddAllDifferent([queens[i] - i for i in range({n})])",
            description="No two queens on the same negative diagonal",
        )
    )

    # Objective: Just find a feasible solution (or all solutions)
    objective = Objective(
        type=ObjectiveType.FEASIBILITY, description="Find feasible queen placement(s)"
    )

    return Problem(
        variables=[queens],
        constraints=constraints,
        objective=objective,
        description=f"{n}-Queens problem: place {n} queens on {n}x{n} board with no attacks",
        parameters={"enumerate_all_solutions": find_all_solutions, "n": n},
    )


def solve_nqueens(n=8, find_all_solutions=False):
    """
    Solve the N-Queens problem and return results.

    Args:
        n: Size of the chessboard
        find_all_solutions: Whether to find all solutions

    Returns:
        dict: Solution results including queen positions and board representation
    """
    problem = create_nqueens_problem(n, find_all_solutions)
    result = solve_problem(problem)

    match result:
        case Success(solution):
            if solution.is_feasible:
                queens_positions = solution.values["queens"]

                # Create board representation
                board = create_board_representation(queens_positions, n)

                # Count attacks (should be 0 for valid solution)
                attack_count = count_attacks(queens_positions)

                return {
                    "status": "feasible",
                    "n": n,
                    "queens": queens_positions,
                    "board": board,
                    "attack_count": attack_count,
                    "is_valid": attack_count == 0,
                    "statistics": solution.statistics,
                }
            else:
                return {
                    "status": solution.status,
                    "error": f"No solution found for {n}-Queens",
                }
        case failure:
            error_msg = (
                failure.failure() if hasattr(failure, "failure") else str(failure)
            )
            return {
                "status": "error",
                "error": f"Failed to solve {n}-Queens problem: {error_msg}",
            }


def create_board_representation(queens, n):
    """Create a visual representation of the chessboard with queens."""
    board = []
    for row in range(n):
        board_row = ["." for _ in range(n)]
        queen_col = queens[row]
        board_row[queen_col] = "Q"
        board.append(board_row)
    return board


def count_attacks(queens):
    """Count the number of attacking pairs of queens."""
    n = len(queens)
    attacks = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Same column
            if queens[i] == queens[j]:
                attacks += 1
            # Same diagonal
            elif abs(queens[i] - queens[j]) == abs(i - j):
                attacks += 1

    return attacks


def print_results(results) -> None:
    """Print N-Queens results in a formatted way."""
    if results["status"] != "feasible":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    n = results["n"]
    queens = results["queens"]
    board = results["board"]

    print(f"{n}-Queens Problem Solution")
    print("=" * (max(30, n * 4)))

    # Print queen positions
    print("\nQueen positions (row, column):")
    for row, col in enumerate(queens):
        print(f"  Row {row}: Column {col}")

    # Print board
    print("\nChessboard:")
    print("  " + " ".join([str(i) for i in range(n)]))
    for row in range(n):
        print(f"{row} " + " ".join(board[row]))

    # Validation
    if results["is_valid"]:
        print("\n✓ Valid solution: No queens attack each other")
    else:
        print(f"\n✗ Invalid solution: {results['attack_count']} attacking pairs")

    # Print solver statistics
    if "statistics" in results:
        print("\nSolver Statistics:")
        for key, value in results["statistics"].items():
            display_key = " ".join(word.title() for word in key.split("_"))
            print(f"  {display_key}: {value}")


def validate_solution(queens) -> bool:
    """Validate that the queens placement is a valid N-Queens solution."""
    n = len(queens)

    # Check that all queens are in different columns
    if len(set(queens)) != n:
        return False

    # Check diagonals
    for i in range(n):
        for j in range(i + 1, n):
            # Check if queens are on the same diagonal
            if abs(queens[i] - queens[j]) == abs(i - j):
                return False

    return True


def benchmark_nqueens(max_n=12) -> None:
    """Benchmark N-Queens for different board sizes."""
    print("N-Queens Benchmark")
    print("=" * 40)
    print("Size | Status    | Time (if available)")
    print("-" * 40)

    for n in range(4, max_n + 1):
        try:
            results = solve_nqueens(n)
            status = "✓ Solved" if results["status"] == "feasible" else "✗ Failed"
            print(f"{n:4d} | {status:9s} | -")
        except Exception as e:
            print(f"{n:4d} | ✗ Error   | {str(e)[:20]}...")


def main() -> None:
    """Main function to run the N-Queens example."""
    print(__doc__)

    # Solve classic 8-Queens
    print("Solving classic 8-Queens problem...")
    results = solve_nqueens(8)
    print_results(results)

    # Optional: benchmark different sizes
    print("\n" + "=" * 50)
    print("Benchmark different board sizes:")
    benchmark_nqueens(10)


def test_nqueens() -> None:
    """Test function for pytest."""
    # Test 4-Queens (smallest non-trivial case)
    results = solve_nqueens(4)
    assert results["status"] == "feasible"
    assert results["n"] == 4
    assert len(results["queens"]) == 4
    assert results["is_valid"]
    assert validate_solution(results["queens"])

    # Test 8-Queens (classic case)
    results = solve_nqueens(8)
    assert results["status"] == "feasible"
    assert results["n"] == 8
    assert len(results["queens"]) == 8
    assert results["is_valid"]
    assert validate_solution(results["queens"])

    # Test that solution has no attacks
    assert results["attack_count"] == 0


if __name__ == "__main__":
    main()

"""
Sparse Matrix Optimization Example

This module demonstrates large-scale optimization problems using sparse matrix format
for memory efficiency. The example solves a resource allocation problem across
multiple facilities and time periods with sparse constraint matrices.

The linear programming problem involves:
- Large number of decision variables (facility-time combinations)
- Sparse constraint matrices (mostly zero coefficients)
- Resource capacity constraints across facilities
- Demand satisfaction constraints across time periods
- Budget and operational constraints

This demonstrates the computational advantage of sparse formats for large problems
where most constraint coefficients are zero.
"""

from returns.result import Success

from usolver_mcp.models.highs_models import (
    HiGHSConstraints,
    HiGHSObjective,
    HiGHSProblem,
    HiGHSProblemSpec,
    HiGHSSparseMatrix,
    HiGHSStatus,
    HiGHSVariable,
)
from usolver_mcp.solvers.highs_solver import solve_problem


def create_sparse_problem():
    """
    Create a large sparse optimization problem.

    Problem structure:
    - 20 facilities x 12 time periods = 240 variables
    - Resource capacity constraints per facility
    - Demand constraints per time period
    - Budget constraints
    - Most constraint coefficients are zero (sparse)

    Returns:
        HiGHSProblem: The sparse optimization problem
    """
    n_facilities = 20
    n_periods = 12
    n_vars = n_facilities * n_periods  # 240 variables

    # Variables: allocation[facility][period]
    variables = []
    for f in range(n_facilities):
        for t in range(n_periods):
            variables.append(
                HiGHSVariable(
                    name=f"alloc_f{f}_t{t}",
                    lb=0.0,  # Non-negative allocations
                    ub=100.0,  # Maximum allocation per facility-period
                    type="cont",
                )
            )

    # Objective: Minimize total cost with varying costs per facility and time
    # Cost increases with facility index and varies by time period
    objective_coeffs = []
    for f in range(n_facilities):
        for t in range(n_periods):
            base_cost = 10 + f * 2  # Facility-dependent base cost
            time_factor = 1 + 0.1 * (t % 4)  # Seasonal variation
            objective_coeffs.append(base_cost * time_factor)

    objective = HiGHSObjective(linear=objective_coeffs)

    # Create sparse constraint matrix
    # We'll have 3 types of constraints:
    # 1. Facility capacity constraints (n_facilities constraints)
    # 2. Time period demand constraints (n_periods constraints)
    # 3. Budget constraint (1 constraint)

    rows = []
    cols = []
    values = []

    constraint_idx = 0

    # 1. Facility capacity constraints
    # Each facility has limited capacity across all time periods
    for f in range(n_facilities):
        800 + f * 50  # Capacity increases with facility size
        for t in range(n_periods):
            var_idx = f * n_periods + t
            rows.append(constraint_idx)
            cols.append(var_idx)
            values.append(1.0)  # Sum of allocations for facility f
        constraint_idx += 1

    # 2. Time period demand constraints
    # Each time period has minimum demand requirements
    for t in range(n_periods):
        300 + t * 25  # Demand increases over time
        for f in range(n_facilities):
            var_idx = f * n_periods + t
            rows.append(constraint_idx)
            cols.append(var_idx)
            values.append(1.0)  # Sum of allocations for time period t
        constraint_idx += 1

    # 3. Budget constraint
    # Total weighted cost cannot exceed budget
    budget_weight_base = 1.5
    for f in range(n_facilities):
        for t in range(n_periods):
            var_idx = f * n_periods + t
            weight = budget_weight_base + f * 0.1 + t * 0.05
            rows.append(constraint_idx)
            cols.append(var_idx)
            values.append(weight)
    constraint_idx += 1

    n_constraints = constraint_idx

    # Create sparse matrix
    sparse_matrix = HiGHSSparseMatrix(
        rows=rows, cols=cols, values=values, shape=[n_constraints, n_vars]
    )

    # Constraint senses and RHS values
    senses = []
    rhs = []

    # Facility capacity constraints (<=)
    for f in range(n_facilities):
        senses.append("<=")
        rhs.append(800 + f * 50)

    # Time period demand constraints (>=)
    for t in range(n_periods):
        senses.append(">=")
        rhs.append(300 + t * 25)

    # Budget constraint (<=)
    senses.append("<=")
    rhs.append(25000)  # Total budget

    constraints = HiGHSConstraints(sparse=sparse_matrix, sense=senses, rhs=rhs)

    problem_spec = HiGHSProblemSpec(
        sense="minimize",
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    return HiGHSProblem(
        problem=problem_spec, options={"output_flag": False, "time_limit": 60.0}
    )


def solve_sparse_optimization():
    """
    Solve the sparse optimization problem and return results.

    Returns:
        dict: Solution results including allocations and analysis
    """
    problem = create_sparse_problem()
    result = solve_problem(problem)

    # Parse the result
    if isinstance(result, Success):
        solution = result.unwrap()
        if solution.status == HiGHSStatus.OPTIMAL:
            return {
                "status": "optimal",
                "objective_value": solution.objective_value,
                "problem_size": {
                    "variables": 240,  # 20 facilities x 12 periods
                    "constraints": 33,  # 20 facility + 12 demand + 1 budget
                    "nonzeros": len(problem.problem.constraints.sparse.values),
                },
            }
        else:
            return {
                "status": "failed",
                "error": f"Non-optimal solution: {solution.status}",
            }
    else:
        return {"status": "error", "error": "Failed to solve problem"}


def analyze_sparsity():
    """Analyze the sparsity pattern of the constraint matrix."""
    problem = create_sparse_problem()
    sparse_matrix = problem.problem.constraints.sparse

    n_rows, n_cols = sparse_matrix.shape
    n_nonzeros = len(sparse_matrix.values)
    total_elements = n_rows * n_cols
    sparsity = 1 - (n_nonzeros / total_elements)

    return {
        "matrix_size": (n_rows, n_cols),
        "total_elements": total_elements,
        "nonzero_elements": n_nonzeros,
        "sparsity_ratio": sparsity,
        "memory_savings": f"{sparsity:.1%}",
    }


def print_results(results) -> None:
    """Print sparse optimization results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Sparse Matrix Optimization Results")
    print("=" * 50)

    problem_size = results["problem_size"]
    print("\nProblem Dimensions:")
    print(f"  Variables: {problem_size['variables']:,}")
    print(f"  Constraints: {problem_size['constraints']:,}")
    print(f"  Non-zero elements: {problem_size['nonzeros']:,}")

    if results["objective_value"] is not None:
        print(f"\nMinimum Total Cost: ${results['objective_value']:,.2f}")

    # Analyze sparsity
    sparsity_info = analyze_sparsity()
    print("\nSparsity Analysis:")
    print(
        f"  Matrix size: {sparsity_info['matrix_size'][0]} x {sparsity_info['matrix_size'][1]}"
    )
    print(f"  Total elements: {sparsity_info['total_elements']:,}")
    print(f"  Non-zero elements: {sparsity_info['nonzero_elements']:,}")
    print(f"  Sparsity ratio: {sparsity_info['sparsity_ratio']:.3f}")
    print(f"  Memory savings: {sparsity_info['memory_savings']}")

    # Calculate density for comparison
    density = 1 - sparsity_info["sparsity_ratio"]
    print("\nMemory Efficiency:")
    if density < 0.1:
        print(f"  ✓ Highly sparse matrix ({density:.1%} density)")
        print("  ✓ Sparse format provides significant memory savings")
    elif density < 0.3:
        print(f"  ✓ Moderately sparse matrix ({density:.1%} density)")
        print("  ✓ Sparse format recommended")
    else:
        print(f"  ⚠ Dense matrix ({density:.1%} density)")
        print("  ⚠ Dense format might be more efficient")


def compare_formats() -> None:
    """Compare memory usage between sparse and dense formats."""
    problem = create_sparse_problem()
    sparse_matrix = problem.problem.constraints.sparse

    n_rows, n_cols = sparse_matrix.shape
    n_nonzeros = len(sparse_matrix.values)

    # Memory estimates (in elements, not bytes)
    sparse_memory = 3 * n_nonzeros  # rows, cols, values arrays
    dense_memory = n_rows * n_cols

    savings_ratio = (dense_memory - sparse_memory) / dense_memory

    print("Memory Format Comparison:")
    print(f"  Dense format: {dense_memory:,} elements")
    print(f"  Sparse format: {sparse_memory:,} elements")
    print(f"  Memory savings: {savings_ratio:.1%}")
    print(f"  Compression ratio: {dense_memory / sparse_memory:.1f}:1")


def main() -> None:
    """Main function to run the sparse optimization example."""
    print(__doc__)

    results = solve_sparse_optimization()
    print_results(results)

    print("\n" + "=" * 50)
    compare_formats()


def test_sparse_optimization() -> None:
    """Test function for pytest."""
    results = solve_sparse_optimization()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test problem dimensions
    problem_size = results["problem_size"]
    assert problem_size["variables"] == 240  # 20 x 12
    assert problem_size["constraints"] == 33  # 20 + 12 + 1
    assert problem_size["nonzeros"] > 0

    # Test that objective value is reasonable
    if results["objective_value"] is not None:
        assert results["objective_value"] > 0

    # Test sparsity analysis
    sparsity_info = analyze_sparsity()
    assert sparsity_info["sparsity_ratio"] > 0.7  # Should be at least 70% sparse
    assert sparsity_info["nonzero_elements"] < sparsity_info["total_elements"]


if __name__ == "__main__":
    main()

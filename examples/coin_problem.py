"""
Coin Problem Logic Puzzle Example

This module solves the classic coin problem logic puzzle using constraint programming.
A friend has 6 US coins totaling $1.15 but cannot make change for various denominations.
The puzzle involves finding which specific coins the friend is holding.

The constraint satisfaction problem involves:
- 6 coins that sum to 115 cents
- Cannot make change for $1.00, $0.50, $0.25, $0.10, or $0.05
- Cannot buy a $0.95 candy bar (excluding half-dollar from purchase)
- All coins must be valid US denominations currently in production

This is a classic logic puzzle suitable for Z3 SMT solving.
"""

from returns.result import Failure, Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.z3_solver import solve_problem


def create_coin_problem():
    """
    Create the coin problem constraint satisfaction problem.

    Returns:
        tuple: (variables, constraints) for the Z3 solver
    """
    # Define variables for each coin (in cents)
    # We'll use 6 integer variables representing the value of each coin
    variables = [
        {"name": "c1", "type": "integer"},
        {"name": "c2", "type": "integer"},
        {"name": "c3", "type": "integer"},
        {"name": "c4", "type": "integer"},
        {"name": "c5", "type": "integer"},
        {"name": "c6", "type": "integer"},
    ]

    constraints = []

    # Valid US coin denominations currently in production (in cents)
    valid_coins = [
        1,
        5,
        10,
        25,
        50,
        100,
    ]  # penny, nickel, dime, quarter, half-dollar, dollar

    # Each coin must be a valid denomination
    for i in range(1, 7):
        coin_constraints = ", ".join([f"c{i} == {coin}" for coin in valid_coins])
        constraints.append(f"Or({coin_constraints})")

    # The sum of all 6 coins equals 115 cents ($1.15)
    constraints.append("c1 + c2 + c3 + c4 + c5 + c6 == 115")

    # Generate all possible subset sums (2 or more coins) and ensure they don't equal forbidden amounts
    forbidden_amounts = [
        100,
        50,
        25,
        10,
        5,
    ]  # dollar, half-dollar, quarter, dime, nickel

    # For each forbidden amount, ensure no subset of 2 or more coins sums to it
    for amount in forbidden_amounts:
        subset_constraints = []

        # All possible subsets of 2 or more coins
        coin_vars = [f"c{i}" for i in range(1, 7)]

        # Generate constraints for all possible subsets
        for mask in range(
            3, 64
        ):  # 3 = 0b000011 (at least 2 coins), 63 = 0b111111 (all 6 coins)
            subset_sum = []
            for i in range(6):
                if mask & (1 << i):
                    subset_sum.append(coin_vars[i])

            if len(subset_sum) >= 2:
                sum_expr = " + ".join(subset_sum)
                subset_constraints.append(f"({sum_expr}) != {amount}")

        # All subsets must not sum to the forbidden amount
        constraints.extend(subset_constraints)

    # Special constraint: cannot buy 95-cent candy bar when half-dollar is excluded
    # This means no subset (excluding any half-dollars) can sum to 95
    vending_constraints = []
    for mask in range(3, 64):  # At least 2 coins
        subset_sum = []
        for i in range(6):
            if mask & (1 << i):
                # Only include if not a half-dollar (50 cents)
                subset_sum.append(f"If(c{i+1} != 50, c{i+1}, 0)")

        if len(subset_sum) >= 2:
            sum_expr = " + ".join(subset_sum)
            vending_constraints.append(f"({sum_expr}) != 95")

    constraints.extend(vending_constraints)

    return variables, constraints


def solve_coin_problem():
    """
    Solve the coin problem and return results.

    Returns:
        dict: Solution results including coin values and analysis
    """
    variables, constraints = create_coin_problem()

    # Convert to Z3Problem model
    z3_variables = [
        Z3Variable(name=var["name"], type=Z3VariableType(var["type"]))
        for var in variables
    ]

    z3_constraints = [Z3Constraint(expression=constraint) for constraint in constraints]

    problem = Z3Problem(
        variables=z3_variables,
        constraints=z3_constraints,
        description="Coin problem logic puzzle",
    )

    result = solve_problem(problem)

    # Parse the result
    match result:
        case Success(solution):
            if solution.is_satisfiable:
                # Extract solution values
                coins = [solution.values[f"c{i}"] for i in range(1, 7)]
                coins.sort(reverse=True)  # Sort in descending order

                return {
                    "status": "satisfiable",
                    "coins": coins,
                    "total_value": sum(coins),
                    "coin_count": len(coins),
                }
            else:
                return {"status": "unsatisfiable", "error": "No solution found"}
        case Failure(error):
            return {"status": "error", "error": str(error)}
        case _:
            return {"status": "error", "error": "Unexpected result type"}


def analyze_solution(results):
    """Analyze the coin solution and verify it meets all constraints."""
    if results["status"] != "satisfiable":
        return None

    coins = results["coins"]

    # Count each denomination
    coin_counts = {}
    for coin in coins:
        coin_counts[coin] = coin_counts.get(coin, 0) + 1

    # Verify total
    total = sum(coins)

    # Test all possible subsets for forbidden amounts
    forbidden_amounts = [100, 50, 25, 10, 5, 95]  # including 95 for vending machine
    forbidden_subsets = []

    for amount in forbidden_amounts:
        for mask in range(3, 64):  # At least 2 coins
            subset = []
            for i in range(6):
                if mask & (1 << i):
                    if (
                        amount == 95 and coins[i] == 50
                    ):  # Exclude half-dollars for vending
                        continue
                    subset.append(coins[i])

            if len(subset) >= 2 and sum(subset) == amount:
                forbidden_subsets.append((amount, subset))

    return {
        "coin_counts": coin_counts,
        "total_cents": total,
        "forbidden_violations": forbidden_subsets,
        "is_valid": len(forbidden_subsets) == 0 and total == 115,
    }


def print_results(results) -> None:
    """Print coin problem results in a formatted way."""
    if results["status"] != "satisfiable":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Coin Problem Logic Puzzle Solution")
    print("=" * 50)

    coins = results["coins"]

    # Map coin values to names
    coin_names = {
        1: "penny",
        5: "nickel",
        10: "dime",
        25: "quarter",
        50: "half-dollar",
        100: "dollar",
    }

    # Count and display coins
    coin_counts = {}
    for coin in coins:
        coin_counts[coin] = coin_counts.get(coin, 0) + 1

    print("\nYour friend has:")
    for value in sorted(coin_counts.keys(), reverse=True):
        count = coin_counts[value]
        name = coin_names.get(value, f"{value}-cent")
        plural = "s" if count > 1 else ""
        print(f"  {count} {name}{plural} ({count} x {value}¢)")

    total_cents = sum(coins)
    print(f"\nTotal: {total_cents}¢ = ${total_cents/100:.2f}")
    print(f"Number of coins: {len(coins)}")

    # Verify the solution
    analysis = analyze_solution(results)
    if analysis and analysis["is_valid"]:
        print("\n✓ Solution verification:")
        print("  - Totals exactly $1.15")
        print("  - Uses exactly 6 coins")
        print("  - Cannot make change for $1.00, $0.50, $0.25, $0.10, or $0.05")
        print("  - Cannot buy $0.95 candy bar (excluding half-dollars)")
    else:
        print("\n✗ Solution verification failed!")
        if analysis and analysis["forbidden_violations"]:
            print("  Forbidden combinations found:")
            for amount, subset in analysis["forbidden_violations"]:
                print(f"    Can make {amount}¢ with: {subset}")


def main() -> None:
    """Main function to run the coin problem example."""
    print(__doc__)

    results = solve_coin_problem()
    print_results(results)


def test_coin_problem() -> None:
    """Test function for pytest."""
    results = solve_coin_problem()

    # Test that we get a satisfiable solution
    assert results["status"] == "satisfiable"

    # Test basic constraints
    assert results["total_value"] == 115  # Must total $1.15
    assert results["coin_count"] == 6  # Must have 6 coins

    # Test that all coins are valid denominations
    valid_coins = {1, 5, 10, 25, 50, 100}
    coins = results["coins"]
    assert all(coin in valid_coins for coin in coins)

    # Test solution analysis
    analysis = analyze_solution(results)
    assert analysis is not None
    assert analysis["is_valid"]
    assert analysis["total_cents"] == 115
    assert len(analysis["forbidden_violations"]) == 0


if __name__ == "__main__":
    main()

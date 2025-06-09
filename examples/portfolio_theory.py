"""
Modern Portfolio Theory Optimization Example

This module demonstrates portfolio optimization using Markowitz mean-variance theory.
The problem involves allocating investment capital across different asset classes
(bonds, stocks, real estate, commodities) to maximize expected return while
constraining portfolio risk and individual asset allocation limits.

The optimization problem minimizes portfolio variance (risk) while achieving
a target expected return, subject to:
- Maximum allocation limits per asset class
- Total allocation must equal 100%
- All allocations must be non-negative
- Portfolio risk cannot exceed a specified threshold

This is a classic convex optimization problem suitable for CVXPY.
"""

import numpy as np
from returns.result import Success

from usolver_mcp.models.cvxpy_models import (
    CVXPYConstraint,
    CVXPYObjective,
    CVXPYProblem,
    CVXPYVariable,
    ObjectiveType,
)
from usolver_mcp.solvers.cvxpy_solver import solve_cvxpy_problem


def create_portfolio_problem():
    """
    Create a modern portfolio theory optimization problem.

    Returns:
        CVXPYProblem: The portfolio optimization problem
    """
    # Asset classes and their properties
    assets = ["Bonds", "Stocks", "RealEstate", "Commodities"]
    n_assets = len(assets)

    # Expected returns (annualized)
    expected_returns = np.array([0.08, 0.12, 0.10, 0.15])

    # Risk factors (standard deviations)
    risk_factors = np.array([0.02, 0.15, 0.08, 0.20])

    # Correlation matrix (simplified)
    correlation_matrix = np.array(
        [
            [1.00, 0.20, 0.30, 0.10],
            [0.20, 1.00, 0.60, 0.70],
            [0.30, 0.60, 1.00, 0.50],
            [0.10, 0.70, 0.50, 1.00],
        ]
    )

    # Covariance matrix
    covariance_matrix = np.outer(risk_factors, risk_factors) * correlation_matrix

    # Define optimization variables
    variables = [CVXPYVariable(name="weights", shape=n_assets)]

    # Define constraints
    constraints = [
        # Budget constraint: weights sum to 1
        CVXPYConstraint(
            expression="cp.sum(weights) == 1",
            description="Total allocation must equal 100%",
        ),
        # Non-negativity constraints
        CVXPYConstraint(
            expression="weights >= 0", description="No short selling allowed"
        ),
        # Individual asset allocation limits
        CVXPYConstraint(
            expression="weights[0] <= 0.4",  # Bonds max 40%
            description="Bonds allocation cannot exceed 40%",
        ),
        CVXPYConstraint(
            expression="weights[1] <= 0.6",  # Stocks max 60%
            description="Stocks allocation cannot exceed 60%",
        ),
        CVXPYConstraint(
            expression="weights[2] <= 0.3",  # Real Estate max 30%
            description="Real Estate allocation cannot exceed 30%",
        ),
        CVXPYConstraint(
            expression="weights[3] <= 0.2",  # Commodities max 20%
            description="Commodities allocation cannot exceed 20%",
        ),
        # Risk constraint: portfolio risk <= 10%
        CVXPYConstraint(
            expression="cp.quad_form(weights, covariance_matrix) <= 0.01",
            description="Portfolio risk cannot exceed 10%",
        ),
    ]

    # Objective: Maximize expected return
    objective = CVXPYObjective(
        type=ObjectiveType.MAXIMIZE, expression="expected_returns.T @ weights"
    )

    return CVXPYProblem(
        variables=variables,
        objective=objective,
        constraints=constraints,
        parameters={
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
        },
        description="Modern Portfolio Theory optimization with risk constraints",
    )


def solve_portfolio_optimization():
    """
    Solve the portfolio optimization problem and return results.

    Returns:
        dict: Solution results including optimal weights and performance metrics
    """
    problem = create_portfolio_problem()
    result = solve_cvxpy_problem(problem)

    match result:
        case Success(solution):
            if solution.status == "optimal":
                weights = solution.values["weights"]
                expected_return = solution.objective_value

                # Calculate portfolio risk
                covariance_matrix = problem.parameters["covariance_matrix"]
                portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)

                return {
                    "status": "optimal",
                    "weights": weights,
                    "expected_return": expected_return,
                    "portfolio_risk": portfolio_risk,
                    "sharpe_ratio": (
                        expected_return / portfolio_risk if portfolio_risk > 0 else 0
                    ),
                }
            else:
                return {"status": solution.status, "error": "No optimal solution found"}
        case _:
            return {"status": "error", "error": "Failed to solve problem"}


def print_results(results) -> None:
    """Print portfolio optimization results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    assets = ["Bonds", "Stocks", "Real Estate", "Commodities"]
    weights = results["weights"]

    print("Portfolio Optimization Results")
    print("=" * 50)
    print("\nOptimal Asset Allocation:")
    for i, asset in enumerate(assets):
        print(f"{asset:15}: {weights[i]:7.1%}")

    print("\nPortfolio Performance:")
    print(f"Expected Return: {results['expected_return']:7.1%} annually")
    print(f"Portfolio Risk:  {results['portfolio_risk']:7.1%} annually")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:7.2f}")

    # Verify constraints
    print("\nConstraint Verification:")
    print(f"Total allocation: {sum(weights):7.1%}")
    print(f"Risk constraint:  {results['portfolio_risk']:7.1%} <= 10.0%")


def main() -> None:
    """Main function to run the portfolio optimization example."""
    print(__doc__)

    results = solve_portfolio_optimization()
    print_results(results)


def test_portfolio_optimization() -> None:
    """Test function for pytest."""
    results = solve_portfolio_optimization()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test that weights sum to approximately 1
    weights = results["weights"]
    assert abs(sum(weights) - 1.0) < 1e-6

    # Test that all weights are non-negative
    assert all(w >= -1e-6 for w in weights)  # Allow small numerical errors

    # Test individual allocation constraints
    assert weights[0] <= 0.4 + 1e-6  # Bonds
    assert weights[1] <= 0.6 + 1e-6  # Stocks
    assert weights[2] <= 0.3 + 1e-6  # Real Estate
    assert weights[3] <= 0.2 + 1e-6  # Commodities

    # Test that portfolio risk is within bounds
    assert results["portfolio_risk"] <= 0.1 + 1e-6


if __name__ == "__main__":
    main()

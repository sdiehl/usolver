from returns.result import Success

from usolver_mcp.models.highs_models import (
    HiGHSConstraints,
    HiGHSConstraintSense,
    HiGHSObjective,
    HiGHSOptions,
    HiGHSProblem,
    HiGHSProblemSpec,
    HiGHSSense,
    HiGHSSparseMatrix,
    HiGHSStatus,
    HiGHSVariable,
    HiGHSVariableType,
)
from usolver_mcp.solvers.highs_solver import solve_problem

pytest_plugins: list[str] = []


def test_simple_linear_program() -> None:
    """Test solving a simple linear program.

    Minimize    f  =  x0 +  x1
    subject to              x1 <= 7
                5 <=  x0 + 2x1 <= 15
                6 <= 3x0 + 2x1
                0 <= x0 <= 4; 1 <= x1
    """
    # Define variables
    variables = [
        HiGHSVariable(name="x0", lb=0, ub=4, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="x1", lb=1, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    # Define objective: minimize x0 + x1
    objective = HiGHSObjective(linear=[1.0, 1.0])

    # Define constraints using dense format
    # x1 <= 7  =>  [0, 1] * [x0, x1] <= 7
    # 5 <= x0 + 2*x1 <= 15  =>  [1, 2] * [x0, x1] >= 5 and [1, 2] * [x0, x1] <= 15
    # 6 <= 3*x0 + 2*x1  =>  [3, 2] * [x0, x1] >= 6
    constraints = HiGHSConstraints(
        dense=[
            [0, 1],  # x1 <= 7
            [1, 2],  # x0 + 2*x1 >= 5
            [1, 2],  # x0 + 2*x1 <= 15
            [3, 2],  # 3*x0 + 2*x1 >= 6
        ],
        sparse=None,
        sense=[
            HiGHSConstraintSense.LESS_EQUAL,  # x1 <= 7
            HiGHSConstraintSense.GREATER_EQUAL,  # x0 + 2*x1 >= 5
            HiGHSConstraintSense.LESS_EQUAL,  # x0 + 2*x1 <= 15
            HiGHSConstraintSense.GREATER_EQUAL,  # 3*x0 + 2*x1 >= 6
        ],
        rhs=[7, 5, 15, 6],
    )

    # Create problem
    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    # Add some options
    options = HiGHSOptions(  # type: ignore[call-arg]
        output_flag=False, log_to_console=False  # Suppress solver output for testing
    )

    problem = HiGHSProblem(problem=problem_spec, options=options)

    # Solve the problem
    result = solve_problem(problem)

    # Check that we got a successful result
    assert isinstance(result, Success)

    solution = result.unwrap()

    # Check that we found an optimal solution
    assert solution.status == HiGHSStatus.OPTIMAL

    # Check that we have solution values for both variables
    assert len(solution.solution) == 2

    # Check that the objective value is reasonable
    # The optimal solution should be around x0=0, x1=2.5 with objective value 2.5
    assert solution.objective_value > 0
    assert solution.objective_value < 10  # Should be a small positive value

    # Check that we have dual information
    assert len(solution.dual_solution) == 4  # One for each constraint
    assert len(solution.variable_duals) == 2  # One for each variable


def test_sparse_constraint_format() -> None:
    """Test solving with sparse constraint format."""
    # Simple problem: minimize x + y subject to x + y >= 1, x >= 0, y >= 0
    variables = [
        HiGHSVariable(name="x", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="y", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    objective = HiGHSObjective(linear=[1.0, 1.0])

    # Use sparse format: x + y >= 1
    sparse_matrix = HiGHSSparseMatrix(
        rows=[0, 0],  # Both coefficients are in row 0
        cols=[0, 1],  # First coefficient is for variable 0, second for variable 1
        values=[1.0, 1.0],  # Both coefficients are 1.0
        shape=(1, 2),  # 1 constraint, 2 variables
    )

    constraints = HiGHSConstraints(
        dense=None,
        sparse=sparse_matrix,
        sense=[HiGHSConstraintSense.GREATER_EQUAL],
        rhs=[1.0],
    )

    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    options = HiGHSOptions(output_flag=False, log_to_console=False)  # type: ignore[call-arg]
    problem = HiGHSProblem(problem=problem_spec, options=options)

    result = solve_problem(problem)

    assert isinstance(result, Success)
    solution = result.unwrap()
    assert solution.status == HiGHSStatus.OPTIMAL

    # Optimal solution should be x=0.5, y=0.5 with objective value 1.0
    assert abs(solution.objective_value - 1.0) < 1e-6


def test_logistics_transportation_problem() -> None:
    """Test solving a logistics transportation problem.

    Problem Summary:
    A company has 2 factories (F1, F2) that produce goods and need to ship them
    to 2 distribution centers (D1, D2), which then distribute to 2 customers (C1, C2).

    - Factory F1 can produce up to 60 units, F2 can produce up to 45 units
    - Customer C1 needs at least 35 units, C2 needs at least 28 units
    - Flow must be conserved at distribution centers (inflow = outflow)
    - Goal: minimize total transportation cost

    Variables represent flow quantities:
    - F1_D1, F1_D2: flow from factory 1 to distribution centers
    - F2_D1, F2_D2: flow from factory 2 to distribution centers
    - D1_C1, D1_C2: flow from distribution center 1 to customers
    - D2_C1, D2_C2: flow from distribution center 2 to customers

    Transportation costs per unit:
    F1→D1: $15.3, F1→D2: $16.8, F2→D1: $14.7, F2→D2: $13.2
    D1→C1: $9.6, D1→C2: $10.4, D2→C1: $11.8, D2→C2: $7.9
    """
    # Define variables for transportation flows
    variables = [
        HiGHSVariable(name="F1_D1", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="F1_D2", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="F2_D1", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="F2_D2", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="D1_C1", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="D1_C2", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="D2_C1", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="D2_C2", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    # Objective: minimize total transportation cost
    objective = HiGHSObjective(linear=[15.3, 16.8, 14.7, 13.2, 9.6, 10.4, 11.8, 7.9])

    # Constraints using dense format
    constraints = HiGHSConstraints(
        dense=[
            # Factory capacity constraints
            [1, 1, 0, 0, 0, 0, 0, 0],  # F1 capacity: F1_D1 + F1_D2 <= 60
            [0, 0, 1, 1, 0, 0, 0, 0],  # F2 capacity: F2_D1 + F2_D2 <= 45
            # Flow conservation at distribution centers
            [1, 0, 1, 0, -1, -1, 0, 0],  # D1: inflow = outflow
            [0, 1, 0, 1, 0, 0, -1, -1],  # D2: inflow = outflow
            # Customer demand constraints
            [0, 0, 0, 0, 1, 0, 1, 0],  # C1 demand: D1_C1 + D2_C1 >= 35
            [0, 0, 0, 0, 0, 1, 0, 1],  # C2 demand: D1_C2 + D2_C2 >= 28
        ],
        sparse=None,
        sense=[
            HiGHSConstraintSense.LESS_EQUAL,  # F1 capacity
            HiGHSConstraintSense.LESS_EQUAL,  # F2 capacity
            HiGHSConstraintSense.EQUAL,  # D1 flow conservation
            HiGHSConstraintSense.EQUAL,  # D2 flow conservation
            HiGHSConstraintSense.GREATER_EQUAL,  # C1 demand
            HiGHSConstraintSense.GREATER_EQUAL,  # C2 demand
        ],
        rhs=[60, 45, 0, 0, 35, 28],  # Capacity, conservation, demand
    )

    # Create problem
    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    # Solver options
    options = HiGHSOptions(output_flag=False, log_to_console=False)  # type: ignore[call-arg]
    problem = HiGHSProblem(problem=problem_spec, options=options)

    # Solve the problem
    result = solve_problem(problem)

    # Verify solution
    assert isinstance(result, Success)
    solution = result.unwrap()
    assert solution.status == HiGHSStatus.OPTIMAL

    # Check that we have solution values for all 8 variables
    assert len(solution.solution) == 8

    # Verify that the solution satisfies basic constraints
    # Variables: F1_D1, F1_D2, F2_D1, F2_D2, D1_C1, D1_C2, D2_C1, D2_C2
    sol = solution.solution

    # Check individual constraint satisfaction
    # Factory capacities
    f1_output = sol[0] + sol[1]  # F1_D1 + F1_D2 <= 60
    f2_output = sol[2] + sol[3]  # F2_D1 + F2_D2 <= 45
    assert f1_output <= 60.0 + 1e-6
    assert f2_output <= 45.0 + 1e-6

    # Customer demands
    c1_received = sol[4] + sol[6]  # D1_C1 + D2_C1 >= 35
    c2_received = sol[5] + sol[7]  # D1_C2 + D2_C2 >= 28
    assert c1_received >= 35.0 - 1e-6
    assert c2_received >= 28.0 - 1e-6

    # Flow conservation at distribution centers
    d1_inflow = sol[0] + sol[2]  # F1_D1 + F2_D1
    d1_outflow = sol[4] + sol[5]  # D1_C1 + D1_C2
    d2_inflow = sol[1] + sol[3]  # F1_D2 + F2_D2
    d2_outflow = sol[6] + sol[7]  # D2_C1 + D2_C2
    assert abs(d1_inflow - d1_outflow) < 1e-6
    assert abs(d2_inflow - d2_outflow) < 1e-6

    # Check that we have dual information for all 6 constraints
    assert len(solution.dual_solution) == 6
    assert len(solution.variable_duals) == 8

    # Objective value should be reasonable (positive, less than naive upper bound)
    assert solution.objective_value > 0
    assert (
        solution.objective_value < 2000
    )  # Should be much less than worst-case routing


def test_markowitz_portfolio_optimization() -> None:
    """Test solving a Markowitz-style portfolio optimization problem.

    Problem Summary:
    An investor wants to allocate $100,000 across 5 different assets to maximize
    expected return while controlling risk through diversification constraints.

    Assets and their expected annual returns:
    - STOCK_A (Tech): 12.5% expected return
    - STOCK_B (Healthcare): 9.8% expected return
    - STOCK_C (Finance): 11.2% expected return
    - BOND_D (Government): 4.3% expected return
    - BOND_E (Corporate): 6.7% expected return

    Constraints:
    - Total investment must equal $100,000
    - No short selling (all weights >= 0)
    - Maximum 40% in any single asset (risk management)
    - At least 20% must be in bonds (conservative requirement)
    - At most 60% in stocks (diversification requirement)

    Goal: Maximize expected portfolio return

    Note: This is a simplified linear version of Markowitz optimization.
    Full Markowitz would include quadratic risk terms (covariance matrix).
    """
    # Define variables for portfolio weights (as dollar amounts)
    variables = [
        HiGHSVariable(name="STOCK_A", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="STOCK_B", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="STOCK_C", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="BOND_D", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
        HiGHSVariable(name="BOND_E", lb=0, ub=None, type=HiGHSVariableType.CONTINUOUS),
    ]

    # Objective: maximize expected return (negate for minimization)
    # Expected returns: 12.5%, 9.8%, 11.2%, 4.3%, 6.7%
    expected_returns = [0.125, 0.098, 0.112, 0.043, 0.067]
    objective = HiGHSObjective(
        linear=[-r for r in expected_returns]
    )  # Negate for maximization

    # Constraints using dense format
    constraints = HiGHSConstraints(
        dense=[
            # Budget constraint: total investment = $100,000
            [1, 1, 1, 1, 1],
            # Individual asset limits: each <= 40% of portfolio ($40,000)
            [1, 0, 0, 0, 0],  # STOCK_A <= 40,000
            [0, 1, 0, 0, 0],  # STOCK_B <= 40,000
            [0, 0, 1, 0, 0],  # STOCK_C <= 40,000
            [0, 0, 0, 1, 0],  # BOND_D <= 40,000
            [0, 0, 0, 0, 1],  # BOND_E <= 40,000
            # At least 20% in bonds ($20,000)
            [0, 0, 0, 1, 1],  # BOND_D + BOND_E >= 20,000
            # At most 60% in stocks ($60,000)
            [1, 1, 1, 0, 0],  # STOCK_A + STOCK_B + STOCK_C <= 60,000
        ],
        sparse=None,
        sense=[
            HiGHSConstraintSense.EQUAL,  # Budget constraint
            HiGHSConstraintSense.LESS_EQUAL,  # STOCK_A limit
            HiGHSConstraintSense.LESS_EQUAL,  # STOCK_B limit
            HiGHSConstraintSense.LESS_EQUAL,  # STOCK_C limit
            HiGHSConstraintSense.LESS_EQUAL,  # BOND_D limit
            HiGHSConstraintSense.LESS_EQUAL,  # BOND_E limit
            HiGHSConstraintSense.GREATER_EQUAL,  # Minimum bonds
            HiGHSConstraintSense.LESS_EQUAL,  # Maximum stocks
        ],
        rhs=[100000, 40000, 40000, 40000, 40000, 40000, 20000, 60000],
    )

    # Create problem (maximize return = minimize negative return)
    problem_spec = HiGHSProblemSpec(
        sense=HiGHSSense.MINIMIZE,  # Minimizing negative return = maximizing return
        objective=objective,
        variables=variables,
        constraints=constraints,
    )

    # Solver options
    options = HiGHSOptions(output_flag=False, log_to_console=False)  # type: ignore[call-arg]
    problem = HiGHSProblem(problem=problem_spec, options=options)

    # Solve the problem
    result = solve_problem(problem)

    # Verify solution
    assert isinstance(result, Success)
    solution = result.unwrap()
    assert solution.status == HiGHSStatus.OPTIMAL

    # Check that we have solution values for all 5 assets
    assert len(solution.solution) == 5

    # Verify portfolio constraints
    sol = solution.solution
    total_investment = sum(sol)
    stock_allocation = sol[0] + sol[1] + sol[2]  # STOCK_A + STOCK_B + STOCK_C
    bond_allocation = sol[3] + sol[4]  # BOND_D + BOND_E

    # Budget constraint
    assert abs(total_investment - 100000.0) < 1e-3

    # Diversification constraints
    assert stock_allocation <= 60000.0 + 1e-3  # Max 60% in stocks
    assert bond_allocation >= 20000.0 - 1e-3  # Min 20% in bonds

    # Individual asset limits (40% max each)
    for allocation in sol:
        assert allocation <= 40000.0 + 1e-3
        assert allocation >= -1e-3  # No short selling

    # Calculate actual expected return
    actual_return = sum(sol[i] * expected_returns[i] for i in range(5))
    expected_return_rate = actual_return / 100000.0  # As percentage

    # Expected return should be reasonable (between bond and stock returns)
    assert expected_return_rate >= 0.043  # At least as good as worst bond
    assert expected_return_rate <= 0.125  # At most as good as best stock

    # Check that we have dual information
    assert len(solution.dual_solution) == 8  # One for each constraint
    assert len(solution.variable_duals) == 5  # One for each asset

    # Objective value should be negative (since we're minimizing negative return)
    assert solution.objective_value <= 0

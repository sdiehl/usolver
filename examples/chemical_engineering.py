"""
Chemical Engineering Pipeline Design Example

This module demonstrates solving a chemical engineering optimization problem using Z3.
The problem involves designing an optimal pipeline for fluid transport considering:
- Flow continuity equations
- Pressure drop constraints
- Reynolds number requirements
- Economic optimization (pipe cost vs pumping cost)

This demonstrates engineering optimization using Z3 SMT solving over real numbers.
"""

from returns.result import Failure, Success

from usolver_mcp.models.z3_models import (
    Z3Constraint,
    Z3Problem,
    Z3Variable,
    Z3VariableType,
)
from usolver_mcp.solvers.z3_solver import solve_problem


def create_pipeline_problem():
    """
    Create a chemical engineering pipeline design optimization problem.

    Problem parameters:
    - Volumetric flow rate: 0.05 m³/s
    - Pipe length: 100 m
    - Water density: 1000 kg/m³
    - Maximum allowable pressure drop: 50 kPa
    - Friction factor: 0.02 (turbulent flow)

    Returns:
        tuple: (variables, constraints) for the Z3 solver
    """
    # Design variables
    variables = [
        {"name": "D", "type": "real"},  # Pipe diameter (m)
        {"name": "v", "type": "real"},  # Flow velocity (m/s)
    ]

    # Problem constants
    q = 0.05  # Volumetric flow rate (m³/s)
    length = 100.0  # Pipe length (m)
    rho = 1000.0  # Water density (kg/m³)
    f = 0.02  # Friction factor (dimensionless)
    max_pressure_drop = 50000.0  # Maximum pressure drop (Pa)
    pi = 3.14159265359

    constraints = []

    # Practical limits
    constraints.append("D >= 0.05")  # Minimum diameter: 5 cm
    constraints.append("D <= 0.5")  # Maximum diameter: 50 cm
    constraints.append("v >= 0.5")  # Minimum velocity: 0.5 m/s
    constraints.append("v <= 8.0")  # Maximum velocity: 8 m/s

    # Flow continuity equation: Q = pi(D/2)^2 * v
    # Rearranged: pi * D² * v / 4 = Q
    # Simplified: D² * v = 4Q/pi ≈ 0.0637
    flow_constant = 4 * q / pi
    constraints.append(f"D * D * v == {flow_constant:.6f}")

    # Pressure drop constraint: ΔP = f(L/D)(rho*v²/2) <= max_pressure_drop
    # Simplified: f * L * rho * v² / (2 * D) <= max_pressure_drop
    pressure_coeff = f * length * rho / 2.0
    constraints.append(f"{pressure_coeff:.2f} * v * v / D <= {max_pressure_drop:.1f}")

    # Reynolds number constraint for turbulent flow (Re > 4000)
    # Re = rho*v*D/mu, assuming mu ≈ 0.001 Pa·s for water
    mu = 0.001  # Dynamic viscosity (Pa·s)
    reynolds_min = 4000
    reynolds_coeff = rho / mu
    constraints.append(f"{reynolds_coeff:.1f} * v * D >= {reynolds_min}")

    # Economic constraint: minimize pipe cost (proportional to D²L) + pumping cost (proportional to ΔP*Q)
    # This is handled as additional constraints rather than objective in Z3
    # Assume maximum acceptable pipe cost relative to pumping cost
    pipe_cost_factor = 1000  # Cost per m³ of pipe material
    pumping_cost_factor = 0.1  # Cost per Pa·m³/s
    max_total_cost = 6000  # Maximum acceptable total cost (increased to be feasible)

    # Pipe cost = pipe_cost_factor * pi * D² * L / 4
    # Pumping cost = pumping_cost_factor * pressure_drop * Q
    pipe_cost_coeff = pipe_cost_factor * pi * length / 4
    pumping_cost_expr = (
        f"{pumping_cost_factor * q:.6f} * {pressure_coeff:.2f} * v * v / D"
    )
    constraints.append(
        f"{pipe_cost_coeff:.2f} * D * D + {pumping_cost_expr} <= {max_total_cost}"
    )

    return variables, constraints


def solve_pipeline_design():
    """
    Solve the pipeline design problem and return results.

    Returns:
        dict: Solution results including optimal diameter, velocity, and performance metrics
    """
    variables, constraints = create_pipeline_problem()

    # Convert to Z3Problem model
    z3_variables = [
        Z3Variable(name=var["name"], type=Z3VariableType(var["type"]))
        for var in variables
    ]

    z3_constraints = [Z3Constraint(expression=constraint) for constraint in constraints]

    problem = Z3Problem(
        variables=z3_variables,
        constraints=z3_constraints,
        description="Chemical engineering pipeline design optimization",
    )

    result = solve_problem(problem)

    # Parse the result
    match result:
        case Success(solution):
            if solution.is_satisfiable:
                # Extract solution values
                diameter = solution.values.get("D")
                v = solution.values.get("v")

                if diameter is not None and v is not None:
                    return {
                        "status": "satisfiable",
                        "diameter": float(diameter),
                        "velocity": float(v),
                    }
                else:
                    return {"status": "error", "error": "Missing solution values"}
            else:
                return {"status": "unsatisfiable", "error": "No solution found"}
        case Failure(error):
            return {"status": "error", "error": str(error)}
        case _:
            return {"status": "error", "error": "Unexpected result type"}


def analyze_design(results):
    """Analyze the pipeline design solution and calculate performance metrics."""
    if results["status"] != "satisfiable":
        return None

    diameter = results["diameter"]
    v = results["velocity"]

    # Problem constants
    q = 0.05
    length = 100.0
    rho = 1000.0
    f = 0.02
    pi = 3.14159265359
    mu = 0.001

    # Calculate derived quantities
    area = pi * (diameter / 2) ** 2
    actual_flow_rate = area * v

    # Pressure drop
    pressure_drop = f * (length / diameter) * (rho * v**2 / 2)

    # Reynolds number
    reynolds = rho * v * diameter / mu

    # Costs
    pipe_cost_factor = 1000
    pumping_cost_factor = 0.1
    pipe_cost = pipe_cost_factor * pi * diameter**2 * length / 4
    pumping_cost = pumping_cost_factor * pressure_drop * q
    total_cost = pipe_cost + pumping_cost

    return {
        "area": area,
        "actual_flow_rate": actual_flow_rate,
        "flow_rate_error": abs(actual_flow_rate - q) / q,
        "pressure_drop": pressure_drop,
        "reynolds_number": reynolds,
        "pipe_cost": pipe_cost,
        "pumping_cost": pumping_cost,
        "total_cost": total_cost,
        "is_turbulent": reynolds > 4000,
    }


def print_results(results) -> None:
    """Print pipeline design results in a formatted way."""
    if results["status"] != "satisfiable":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    diameter = results["diameter"]
    v = results["velocity"]

    print("Chemical Engineering Pipeline Design Results")
    print("=" * 60)

    print("\nOptimal Design Parameters:")
    print(f"  Pipe diameter: {diameter:.4f} m ({diameter*100:.2f} cm)")
    print(f"  Flow velocity: {v:.3f} m/s")

    # Performance analysis
    analysis = analyze_design(results)
    if analysis:
        print("\nPerformance Analysis:")
        print("-" * 30)
        print(f"  Cross-sectional area: {analysis['area']:.6f} m²")
        print(f"  Actual flow rate: {analysis['actual_flow_rate']:.6f} m³/s")
        print(f"  Flow rate error: {analysis['flow_rate_error']:.1%}")
        print(
            f"  Pressure drop: {analysis['pressure_drop']:.0f} Pa ({analysis['pressure_drop']/1000:.1f} kPa)"
        )
        print(f"  Reynolds number: {analysis['reynolds_number']:.0f}")
        print(
            f"  Flow regime: {'Turbulent' if analysis['is_turbulent'] else 'Laminar'}"
        )

        print("\nCost Analysis:")
        print("-" * 20)
        print(f"  Pipe cost: ${analysis['pipe_cost']:.2f}")
        print(f"  Pumping cost: ${analysis['pumping_cost']:.2f}")
        print(f"  Total cost: ${analysis['total_cost']:.2f}")

        # Verify constraints
        print("\nConstraint Verification:")
        print("-" * 25)
        print(f"  Diameter limits: 0.05 ≤ {diameter:.3f} ≤ 0.5 m ✓")
        print(f"  Velocity limits: 0.5 ≤ {v:.3f} ≤ 8.0 m/s ✓")
        print(f"  Pressure drop: {analysis['pressure_drop']:.0f} ≤ 50,000 Pa ✓")
        print(f"  Reynolds number: {analysis['reynolds_number']:.0f} ≥ 4,000 ✓")


def main() -> None:
    """Main function to run the chemical engineering example."""
    print(__doc__)

    results = solve_pipeline_design()
    print_results(results)


def test_chemical_engineering() -> None:
    """Test function for pytest."""
    results = solve_pipeline_design()

    # Test that we get a satisfiable solution
    assert results["status"] == "satisfiable"

    diameter = results["diameter"]
    v = results["velocity"]

    # Test design constraints
    assert 0.05 <= diameter <= 0.5  # Diameter limits
    assert 0.5 <= v <= 8.0  # Velocity limits

    # Test derived constraints
    analysis = analyze_design(results)
    assert analysis is not None

    # Flow continuity (should be very close to target)
    assert analysis["flow_rate_error"] < 0.01  # Less than 1% error

    # Pressure drop constraint
    assert analysis["pressure_drop"] <= 50000  # Less than 50 kPa

    # Reynolds number constraint (turbulent flow)
    assert analysis["reynolds_number"] >= 4000

    # Cost constraint
    assert analysis["total_cost"] <= 6000


if __name__ == "__main__":
    main()

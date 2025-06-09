"""
Transportation/Logistics Optimization Example

This module demonstrates transportation network optimization for a supply chain.
The problem involves minimizing transportation costs while meeting supply and demand
constraints across a network of suppliers, warehouses, and customers.

The linear programming problem minimizes total transportation cost subject to:
- Supply constraints at suppliers (cannot exceed available supply)
- Demand constraints at customers (must meet minimum demand)
- Flow conservation at warehouses (inflow = outflow)
- Non-negativity constraints on all flows
- Optional capacity constraints on transportation routes

This is a classic transportation problem suitable for HiGHS linear programming.
"""

from usolver_mcp.solvers.highs_solver import simple_highs_solver


def create_logistics_problem():
    """
    Create a transportation/logistics optimization problem.

    Network structure:
    - 2 Suppliers (S1, S2) with supply capacities
    - 2 Warehouses (W1, W2) acting as transshipment points
    - 2 Customers (C1, C2) with demand requirements

    Returns:
        dict: Problem parameters for the HiGHS solver
    """
    # Decision variables (flows):
    # S1->W1, S1->W2, S2->W1, S2->W2  (supplier to warehouse)
    # W1->C1, W1->C2, W2->C1, W2->C2  (warehouse to customer)

    # Transportation costs per unit
    costs = [
        12.5,  # S1 -> W1
        14.2,  # S1 -> W2
        13.8,  # S2 -> W1
        11.9,  # S2 -> W2
        8.4,  # W1 -> C1
        9.1,  # W1 -> C2
        10.5,  # W2 -> C1
        6.2,  # W2 -> C2
    ]

    # Variable definitions (all continuous, non-negative flows)
    variables = [
        {"name": "S1_W1", "lb": 0, "type": "cont"},  # Supplier 1 to Warehouse 1
        {"name": "S1_W2", "lb": 0, "type": "cont"},  # Supplier 1 to Warehouse 2
        {"name": "S2_W1", "lb": 0, "type": "cont"},  # Supplier 2 to Warehouse 1
        {"name": "S2_W2", "lb": 0, "type": "cont"},  # Supplier 2 to Warehouse 2
        {"name": "W1_C1", "lb": 0, "type": "cont"},  # Warehouse 1 to Customer 1
        {"name": "W1_C2", "lb": 0, "type": "cont"},  # Warehouse 1 to Customer 2
        {"name": "W2_C1", "lb": 0, "type": "cont"},  # Warehouse 2 to Customer 1
        {"name": "W2_C2", "lb": 0, "type": "cont"},  # Warehouse 2 to Customer 2
    ]

    # Constraint matrix
    # Rows: [S1_supply, S2_supply, W1_flow_conservation, W2_flow_conservation, C1_demand, C2_demand]
    # Cols: [S1_W1, S1_W2, S2_W1, S2_W2, W1_C1, W1_C2, W2_C1, W2_C2]
    constraint_matrix = [
        # Supply constraints
        [1, 1, 0, 0, 0, 0, 0, 0],  # S1 supply: S1_W1 + S1_W2 <= 50
        [0, 0, 1, 1, 0, 0, 0, 0],  # S2 supply: S2_W1 + S2_W2 <= 40
        # Flow conservation at warehouses (inflow = outflow)
        [1, 0, 1, 0, -1, -1, 0, 0],  # W1: S1_W1 + S2_W1 = W1_C1 + W1_C2
        [0, 1, 0, 1, 0, 0, -1, -1],  # W2: S1_W2 + S2_W2 = W2_C1 + W2_C2
        # Demand constraints
        [0, 0, 0, 0, 1, 0, 1, 0],  # C1 demand: W1_C1 + W2_C1 >= 30
        [0, 0, 0, 0, 0, 1, 0, 1],  # C2 demand: W1_C2 + W2_C2 >= 25
    ]

    # Constraint senses and right-hand sides
    constraint_senses = ["<=", "<=", "=", "=", ">=", ">="]
    rhs_values = [
        50,
        40,
        0,
        0,
        30,
        25,
    ]  # Supply limits, flow conservation, demand requirements

    return {
        "sense": "minimize",
        "objective_coeffs": costs,
        "variables": variables,
        "constraint_matrix": constraint_matrix,
        "constraint_senses": constraint_senses,
        "rhs_values": rhs_values,
        "description": "Transportation network optimization to minimize shipping costs",
    }


def solve_logistics_optimization():
    """
    Solve the logistics optimization problem and return results.

    Returns:
        dict: Solution results including optimal flows and total cost
    """
    problem_params = create_logistics_problem()
    result = simple_highs_solver(**problem_params)

    # Parse the result from the HiGHS solver
    if result and len(result) > 0:
        result_text = result[0].text

        # Extract solution status and values
        if "optimal" in result_text.lower():
            lines = result_text.split("\n")
            solution_data = {}

            # Parse variable values
            flow_vars = [
                "S1_W1",
                "S1_W2",
                "S2_W1",
                "S2_W2",
                "W1_C1",
                "W1_C2",
                "W2_C1",
                "W2_C2",
            ]
            for line in lines:
                for var in flow_vars:
                    if f"{var}:" in line:
                        try:
                            solution_data[var] = float(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            continue

                # Extract objective value
                if "Objective value:" in line:
                    try:
                        solution_data["objective_value"] = float(
                            line.split(":")[1].strip()
                        )
                    except (ValueError, IndexError):
                        continue

            if solution_data:
                return {
                    "status": "optimal",
                    "flows": solution_data,
                    "total_cost": solution_data.get("objective_value", 0),
                }

    return {"status": "error", "error": "Failed to solve problem"}


def analyze_solution(results, problem_params):
    """Analyze the logistics solution and verify constraints."""
    if results["status"] != "optimal":
        return None

    flows = results["flows"]

    # Calculate supply utilization
    s1_used = flows.get("S1_W1", 0) + flows.get("S1_W2", 0)
    s2_used = flows.get("S2_W1", 0) + flows.get("S2_W2", 0)

    # Calculate demand satisfaction
    c1_satisfied = flows.get("W1_C1", 0) + flows.get("W2_C1", 0)
    c2_satisfied = flows.get("W1_C2", 0) + flows.get("W2_C2", 0)

    # Calculate warehouse throughput
    w1_inflow = flows.get("S1_W1", 0) + flows.get("S2_W1", 0)
    w1_outflow = flows.get("W1_C1", 0) + flows.get("W1_C2", 0)
    w2_inflow = flows.get("S1_W2", 0) + flows.get("S2_W2", 0)
    w2_outflow = flows.get("W2_C1", 0) + flows.get("W2_C2", 0)

    return {
        "supply_utilization": {
            "S1": {"used": s1_used, "capacity": 50, "utilization": s1_used / 50},
            "S2": {"used": s2_used, "capacity": 40, "utilization": s2_used / 40},
        },
        "demand_satisfaction": {
            "C1": {"satisfied": c1_satisfied, "required": 30},
            "C2": {"satisfied": c2_satisfied, "required": 25},
        },
        "warehouse_flows": {
            "W1": {
                "inflow": w1_inflow,
                "outflow": w1_outflow,
                "balanced": abs(w1_inflow - w1_outflow) < 1e-6,
            },
            "W2": {
                "inflow": w2_inflow,
                "outflow": w2_outflow,
                "balanced": abs(w2_inflow - w2_outflow) < 1e-6,
            },
        },
    }


def print_results(results, problem_params):
    """Print logistics optimization results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    flows = results["flows"]

    print("Transportation/Logistics Optimization Results")
    print("=" * 60)
    print(f"\nMinimum Total Transportation Cost: ${results['total_cost']:,.2f}")

    # Print optimal flows
    print("\nOptimal Flow Distribution:")
    print("-" * 40)

    print("Supplier to Warehouse flows:")
    print(f"  S1 -> W1: {flows.get('S1_W1', 0):6.2f} units")
    print(f"  S1 -> W2: {flows.get('S1_W2', 0):6.2f} units")
    print(f"  S2 -> W1: {flows.get('S2_W1', 0):6.2f} units")
    print(f"  S2 -> W2: {flows.get('S2_W2', 0):6.2f} units")

    print("\nWarehouse to Customer flows:")
    print(f"  W1 -> C1: {flows.get('W1_C1', 0):6.2f} units")
    print(f"  W1 -> C2: {flows.get('W1_C2', 0):6.2f} units")
    print(f"  W2 -> C1: {flows.get('W2_C1', 0):6.2f} units")
    print(f"  W2 -> C2: {flows.get('W2_C2', 0):6.2f} units")

    # Analysis
    analysis = analyze_solution(results, problem_params)
    if analysis:
        print("\nSupply Chain Analysis:")
        print("-" * 30)

        supply_util = analysis["supply_utilization"]
        print("Supply Utilization:")
        for supplier, data in supply_util.items():
            print(
                f"  {supplier}: {data['used']:6.2f} / {data['capacity']:6.2f} ({data['utilization']:6.1%})"
            )

        demand_sat = analysis["demand_satisfaction"]
        print("\nDemand Satisfaction:")
        for customer, data in demand_sat.items():
            status = "✓" if data["satisfied"] >= data["required"] else "✗"
            print(
                f"  {customer}: {data['satisfied']:6.2f} / {data['required']:6.2f} {status}"
            )

        warehouse_flows = analysis["warehouse_flows"]
        print("\nWarehouse Flow Balance:")
        for warehouse, data in warehouse_flows.items():
            status = "✓" if data["balanced"] else "✗"
            print(
                f"  {warehouse}: In={data['inflow']:6.2f}, Out={data['outflow']:6.2f} {status}"
            )


def main():
    """Main function to run the logistics optimization example."""
    print(__doc__)

    problem_params = create_logistics_problem()
    results = solve_logistics_optimization()
    print_results(results, problem_params)


def test_logistics():
    """Test function for pytest."""
    problem_params = create_logistics_problem()
    results = solve_logistics_optimization()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test that all flows are non-negative
    flows = results["flows"]
    flow_vars = ["S1_W1", "S1_W2", "S2_W1", "S2_W2", "W1_C1", "W1_C2", "W2_C1", "W2_C2"]
    for var in flow_vars:
        if var in flows:
            assert flows[var] >= -1e-6  # Allow small numerical errors

    # Test supply constraints
    s1_total = flows.get("S1_W1", 0) + flows.get("S1_W2", 0)
    s2_total = flows.get("S2_W1", 0) + flows.get("S2_W2", 0)
    assert s1_total <= 50 + 1e-6  # S1 supply constraint
    assert s2_total <= 40 + 1e-6  # S2 supply constraint

    # Test demand constraints
    c1_total = flows.get("W1_C1", 0) + flows.get("W2_C1", 0)
    c2_total = flows.get("W1_C2", 0) + flows.get("W2_C2", 0)
    assert c1_total >= 30 - 1e-6  # C1 demand constraint
    assert c2_total >= 25 - 1e-6  # C2 demand constraint

    # Test flow conservation at warehouses
    w1_in = flows.get("S1_W1", 0) + flows.get("S2_W1", 0)
    w1_out = flows.get("W1_C1", 0) + flows.get("W1_C2", 0)
    w2_in = flows.get("S1_W2", 0) + flows.get("S2_W2", 0)
    w2_out = flows.get("W2_C1", 0) + flows.get("W2_C2", 0)

    assert abs(w1_in - w1_out) < 1e-6  # W1 flow conservation
    assert abs(w2_in - w2_out) < 1e-6  # W2 flow conservation

    # Test that total cost is positive
    assert results["total_cost"] > 0


if __name__ == "__main__":
    main()

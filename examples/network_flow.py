"""
Network Flow Optimization Example

This module demonstrates network flow optimization for finding shortest paths in directed graphs.
The problem involves finding the minimum-cost path from a source node to a destination node
through a network with weighted edges.

The linear programming problem involves:
- Binary decision variables for edge selection (use edge or not)
- Flow conservation constraints at each node
- Minimize total path cost objective

This is a classic shortest path problem suitable for HiGHS linear programming.
"""

from returns.result import Failure, Success

from usolver_mcp.solvers.highs_solver import simple_highs_solver


def create_network_flow_problem():
    """
    Create a network flow optimization problem for shortest path.

    The problem involves finding the shortest path from node 'A' to node 'D'
    through a directed graph with weighted edges representing distances/costs.

    Graph structure:
        A --2.0--> B --2.5--> D
        |          |          ^
        1.5        3.0        1.0
        |          |          |
        v          v          |
        C --------------------

    Returns:
        dict: Problem parameters for the HiGHS solver
    """
    # Define nodes
    nodes = ["A", "B", "C", "D"]

    # Define edges as (from_node, to_node, weight)
    edges = [
        ("A", "B", 2.0),
        ("A", "C", 1.5),
        ("B", "C", 3.0),
        ("B", "D", 2.5),
        ("C", "D", 1.0),
    ]

    # Source and destination
    source = "A"
    destination = "D"

    n_edges = len(edges)
    edge_costs = [edge[2] for edge in edges]

    # Variable definitions (binary: use edge or not)
    variables = []
    for from_node, to_node, _ in edges:
        variables.append(
            {
                "name": f"edge_{from_node}_{to_node}",
                "lb": 0,
                "ub": 1,
                "type": "bin",  # Binary variable
            }
        )

    # Flow conservation constraints
    # For each node: sum(incoming flow) - sum(outgoing flow) = net_supply
    # net_supply = 1 for source, -1 for destination, 0 for intermediate nodes

    constraint_matrix = []
    constraint_senses = []
    rhs_values = []

    for node in nodes:
        # Create constraint for this node
        constraint_row = [0] * n_edges

        for i, (from_node, to_node, _) in enumerate(edges):
            if from_node == node:
                constraint_row[i] = -1  # Outgoing flow (negative)
            elif to_node == node:
                constraint_row[i] = 1  # Incoming flow (positive)

        constraint_matrix.append(constraint_row)
        constraint_senses.append("=")

        # Set RHS based on node type
        if node == source:
            rhs_values.append(-1)  # Source supplies 1 unit of flow
        elif node == destination:
            rhs_values.append(1)  # Destination consumes 1 unit of flow
        else:
            rhs_values.append(0)  # Intermediate nodes have zero net flow

    return {
        "sense": "minimize",
        "objective_coeffs": edge_costs,
        "variables": variables,
        "constraint_matrix": constraint_matrix,
        "constraint_senses": constraint_senses,
        "rhs_values": rhs_values,
        "description": "Network flow optimization for shortest path finding",
        "nodes": nodes,
        "edges": edges,
        "source": source,
        "destination": destination,
    }


def solve_network_flow():
    """
    Solve the network flow problem and return results.

    Returns:
        dict: Solution results including selected edges and total cost
    """
    problem_params = create_network_flow_problem()

    # Extract solver parameters (exclude metadata)
    solver_params = {
        k: v
        for k, v in problem_params.items()
        if k not in ["nodes", "edges", "source", "destination"]
    }
    result = simple_highs_solver(**solver_params)

    # Parse the result from the HiGHS solver
    match result:
        case Success(solution):
            if solution.status.value == "optimal":
                # Extract solution values
                solution_values = solution.solution or []

                # Map solution values to edge selections
                selections = {}
                for i in range(len(problem_params["edges"])):
                    from_node, to_node, _ = problem_params["edges"][i]
                    var_name = f"edge_{from_node}_{to_node}"
                    if i < len(solution_values):
                        selections[var_name] = round(
                            solution_values[i]
                        )  # Binary variable
                    else:
                        selections[var_name] = 0

                return {
                    "status": "optimal",
                    "selections": selections,
                    "total_cost": solution.objective_value or 0,
                }
            else:
                return {
                    "status": solution.status.value,
                    "error": f"Problem status: {solution.status.value}",
                }
        case Failure(error):
            return {"status": "error", "error": str(error)}
        case _:
            return {"status": "error", "error": "Unexpected result type"}


def analyze_solution(results, problem_params):
    """Analyze the network flow solution to extract the path."""
    if results["status"] != "optimal":
        return None

    edges = problem_params["edges"]
    selections = results["selections"]
    source = problem_params["source"]
    destination = problem_params["destination"]

    # Find selected edges
    selected_edges = []
    for from_node, to_node, cost in edges:
        var_name = f"edge_{from_node}_{to_node}"
        if selections.get(var_name, 0) == 1:
            selected_edges.append((from_node, to_node, cost))

    # Reconstruct path
    path = [source]
    current_node = source

    while current_node != destination:
        # Find the next node in the path
        next_node = None
        for from_node, to_node, _ in selected_edges:
            if from_node == current_node:
                next_node = to_node
                break

        if next_node is None:
            return None  # Path not found (shouldn't happen with valid solution)

        path.append(next_node)
        current_node = next_node

    return {
        "path": path,
        "selected_edges": selected_edges,
        "total_cost": results["total_cost"],
    }


def print_results(results, problem_params) -> None:
    """Print network flow results in a formatted way."""
    if results["status"] != "optimal":
        print(f"Problem Status: {results['status']}")
        if "error" in results:
            print(f"Error: {results['error']}")
        return

    print("Network Flow Optimization Results")
    print("=" * 50)

    # Analyze solution
    analysis = analyze_solution(results, problem_params)
    if not analysis:
        print("Could not reconstruct path from solution")
        return

    source = problem_params["source"]
    destination = problem_params["destination"]

    print(f"\nShortest path from {source} to {destination}:")
    print(f"Path: {' -> '.join(analysis['path'])}")
    print(f"Total cost: {analysis['total_cost']}")

    print("\nSelected edges:")
    print("-" * 30)
    for from_node, to_node, cost in analysis["selected_edges"]:
        print(f"  {from_node} -> {to_node} (cost: {cost})")

    # Show graph structure
    print("\nGraph structure:")
    print("-" * 20)
    edges = problem_params["edges"]
    for from_node, to_node, cost in edges:
        selected = (from_node, to_node, cost) in analysis["selected_edges"]
        status = "SELECTED" if selected else "not used"
        print(f"  {from_node} -> {to_node} (cost: {cost}) [{status}]")


def main() -> None:
    """Main function to run the network flow example."""
    print(__doc__)

    problem_params = create_network_flow_problem()
    results = solve_network_flow()
    print_results(results, problem_params)


def test_network_flow() -> None:
    """Test function for pytest."""
    problem_params = create_network_flow_problem()
    results = solve_network_flow()

    # Test that we get an optimal solution
    assert results["status"] == "optimal"

    # Test solution analysis
    analysis = analyze_solution(results, problem_params)
    assert analysis is not None

    # Test that path starts with source and ends with destination
    path = analysis["path"]
    assert path[0] == problem_params["source"]
    assert path[-1] == problem_params["destination"]

    # Test that path is continuous (each step uses a valid edge)
    edges = problem_params["edges"]
    edge_dict = {(from_node, to_node): cost for from_node, to_node, cost in edges}

    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        assert (
            from_node,
            to_node,
        ) in edge_dict, f"Invalid edge in path: {from_node} -> {to_node}"

    # Test that total cost is positive
    assert results["total_cost"] > 0

    # Test that the known optimal path is found (A -> C -> D with cost 2.5)
    expected_cost = 2.5  # 1.5 (A->C) + 1.0 (C->D)
    assert abs(results["total_cost"] - expected_cost) < 1e-6

    # Test that the expected path is found
    expected_path = ["A", "C", "D"]
    assert path == expected_path


if __name__ == "__main__":
    main()

<p align="center">
    <img src=".github/logo.png" width="500px" alt="usolver">
</p>

# USolver

A best-effort universal logic and numerical solver interface using MCP. Implements the "LLM sandwich" model where a query is interpreted by the LLM, calls out to a dedicated efficient solver fit for the problem, and then verbalizes the result. And the solver solutions can be chained together to solve more complex problems that require multi-step approaches.

Exposes minimal solvers for the following software packages:

* `ortools` - Combinatorial optimization solver
* `cvxpy` - Convex optimization solver
* `z3` - SMT solver over booleans, integers, reals, and strings

To install run the `install.py` script. This will install the MPC server for Claude Desktop and/or Cursor.

```shell
uv run install.py
```

## Examples

To run the individual solver examples. You can invoke the individual examples. Below are example prompts that you can feed to the language model for these specific problems.

```shell
uv run examples/example_z3.py
uv run examples/example_cvxpy.py
uv run examples/example_ortools.py
uv run examples/example_z3_simple.py
```

### Z3

A chemical engineering example:

```markdown
Use usolver to design a water transport pipeline with the following requirements:

* Volumetric flow rate: 0.05 m³/s
* Pipe length: 100 m
* Water density: 1000 kg/m³
* Maximum allowable pressure drop: 50 kPa
* Flow continuity: Q = π(D/2)² × v
* Pressure drop: ΔP = f(L/D)(ρv²/2), where f ≈ 0.02 for turbulent flow
* Practical limits: 0.05 ≤ D ≤ 0.5 m, 0.5 ≤ v ≤ 8 m/s
* Pressure constraint: ΔP ≤ 50,000 Pa
* Find: optimal pipe diameter and flow velocity
```

### CVXPY

A simple convex optimization problem minimizing the 2-norm of a linear system:

```markdown
Use usolver to solve the following convex optimization problem:

Minimize: ||Ax - b||₂²
Subject to: 0 ≤ x ≤ 1
where 
  A = [1.0, -0.5; 0.5, 2.0; 0.0, 1.0] 
  b = [2.0, 1.0, -1.0]
```

### OR-Tools

A classic worker shift scheduling problem:

```markdown
Use usolver to solve a nurse scheduling problem with the following requirements:

* Schedule 4 nurses (Alice, Bob, Charlie, Diana) across 3 shifts over (Monday, Tuesday, Wednesday)
* Shifts: Morning (7AM-3PM), Evening (3PM-11PM), Night (11PM-7AM)
* Each shift must be assigned to exactly one nurse each day
* Each nurse works at most one shift per day
* Distribute shifts evenly (2-3 shifts per nurse over the period)
* Charlie can't work on Tuesday.
```

### Chained Examples

A chained example that uses both OR-Tools to optimize for table layout and CVXPY to optimize for staff scheduling.

```markdown
Use usolver to optimize a restaurant's layout and staffing with the following requirements in two parts. Use combinatorial optimization to optimize for table layout and convex optimization to optimize for staff scheduling.

* Part 1: Optimize table layout
  - Mix of 2-seater, 4-seater, and 6-seater tables
  - Maximum floor space: 150 m²
  - Space requirements: 4m² (2-seater), 6m² (4-seater), 9m² (6-seater)
  - Maximum 20 tables total
  - Minimum mix: 2× 2-seaters, 3× 4-seaters, 1× 6-seater
  - Objective: Maximize total seating capacity

* Part 2: Optimize staff scheduling using Part 1's capacity
  - 12-hour operating day
  - Each staff member can handle 20 seats
  - Minimum 2 staff per hour
  - Maximum staff change between hours: 2 people
  - Variable demand: 40%-100% of capacity
  - Objective: Minimize labor cost ($25/hour per staff)
```

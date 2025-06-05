<p align="center">
    <img src=".github/logo.png" width="500px" alt="usolver">
</p>

# USolver

A Model Context Protocol server that exposes tools for solving combinatorial, convex, integer programming, and non-linear optimization problems. Exposes interfaces to the following solvers:

* [`ortools`](https://developers.google.com/optimization) - Combinatorial optimization solver
* [`cvxpy`](https://www.cvxpy.org/) - Convex optimization solver
* [`z3`](https://github.com/Z3Prover/z3) - SMT solver over booleans, integers, reals, and strings

To install run the `install.py` script. This will install the MPC server for Claude Desktop and/or Cursor.

```shell
uv run install.py
```

Then open Claude or Cursor and you should see the MCP tool `usolver` available in the tool list.

## Examples

To run the individual solver examples. You can invoke the individual examples. Below are example prompts that you can feed to the language model for these specific problems.

```shell
uv run examples/example_z3.py
uv run examples/example_cvxpy.py
uv run examples/example_ortools.py
uv run examples/example_z3_simple.py
```

### Logic Puzzle

```
You and a friend pass by a standard coin operated vending machine and you decide to get a candy bar.
The price is US $0.95, but after checking your pockets you only have a dollar (US $1) and the machine
only takes coins. You turn to your friend and have this conversation:

You: Hey, do you have change for a dollar?
Friend: Let's see. I have 6 US coins but, although they add up to a US $1.15, I can't break a dollar.
You: Huh? Can you make change for half a dollar?
Friend: No.
You: How about a quarter?
Friend: Nope, and before you ask I cant make change for a dime or nickel either.
You: Really? and these six coins are all US government coins currently in production?
Friend: Yes.
You: Well can you just put your coins into the vending machine and buy me a candy bar, and I'll pay you back?
Friend: Sorry, I would like to but I can't with the coins I have.

What coins are your friend holding?
```

This can be fed into usolver and it will generate a constraint system:

$C$ is the collection of the six unknown coin values, $c_1$ through $c_6$, each of which must be a positive whole number representing cents.

$$
C = \{c_1, c_2, c_3, c_4, c_5, c_6\}, \quad \text{where each } c_i \in \mathbb{Z}^+
$$

$\mathcal{S}$ is the collection of every possible way you could choose two or more of your six coins.

$$
\mathcal{S} = \\{S \mid S \subseteq C \land |S| \ge 2 \\}
$$

$$
v(x) = \begin{cases} 0 & \text{if } x = 50 \\\\ x & \text{if } x \neq 50 \end{cases}
$$

Constraint 0: The sum of the values of all six coins is 115 cents.

$$
\sum_{i=1}^{6} c_i = 115
$$

Constraint 1: Cannot make change for a dollar.

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} x \neq 100
$$

Constraint 2: Cannot make change for half a dollar.

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} x \neq 50
$$

Constraint 3: Cannot make change for a quarter.

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} x \neq 25
$$

Constraint 4: Cannot make change for a dime.

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} x \neq 10
$$

Constraint 5: Cannot make change for a nickel

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} x \neq 5
$$

Constraint 6: Cannot buy the candy because the vending machines does not take 50 cent coins.

$$
\forall S \in \mathcal{S}, \quad \sum_{x \in S} v(x) \neq 95
$$

If you feed this to solver it will synthesize the above constraint system, solve it with Z3, and return the solution.

```markdown
Your friend has: 1 half dollar, 1 quarter, and 4 dimes
This totals 50¢ + 25¢ + 40¢ = 115¢ = $1.15 ✓
This is exactly 6 coins ✓
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

## Docker Usage

Can also run the MCP server directly from the GitHub Container Registry.

```bash
docker run -p 8081:8081 ghcr.io/sdiehl/usolver:latest
```

Then add the following to your client:

```json
{
  "mcpServers": {
    "sympy-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "-p",
        "8081:8081",
        "--rm",
        "ghcr.io/sdiehl/usolver:latest"
      ]
    }
  }
}
```

## License

Released under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

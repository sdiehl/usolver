from enum import Enum

from pydantic import BaseModel, Field


class HiGHSSense(str, Enum):
    """Optimization sense for HiGHs problems."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class HiGHSVariableType(str, Enum):
    """Variable types in HiGHs."""

    CONTINUOUS = "cont"
    INTEGER = "int"
    BINARY = "bin"


class HiGHSConstraintSense(str, Enum):
    """Constraint directions in HiGHs."""

    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "="


class HiGHSPresolve(str, Enum):
    """Presolve options for HiGHs."""

    OFF = "off"
    CHOOSE = "choose"
    ON = "on"


class HiGHSSolver(str, Enum):
    """Solver options for HiGHs."""

    SIMPLEX = "simplex"
    CHOOSE = "choose"
    IPM = "ipm"
    PDLP = "pdlp"


class HiGHSParallel(str, Enum):
    """Parallel options for HiGHs."""

    OFF = "off"
    CHOOSE = "choose"
    ON = "on"


class HiGHSStatus(str, Enum):
    """Solver status values for HiGHs."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"
    UNKNOWN = "unknown"


class HiGHSObjective(BaseModel):
    """Objective function specification for HiGHs."""

    linear: list[float] = Field(..., description="Coefficients for each variable")


class HiGHSVariable(BaseModel):
    """Variable specification for HiGHs."""

    name: str | None = Field(
        None, description="Variable name (optional, defaults to x1, x2, etc.)"
    )
    lb: float | None = Field(None, description="Lower bound (optional, defaults to 0)")
    ub: float | None = Field(
        None, description="Upper bound (optional, defaults to +∞, except binary gets 1)"
    )
    type: HiGHSVariableType | None = Field(
        None, description="Variable type (optional, defaults to 'cont')"
    )


class HiGHSSparseMatrix(BaseModel):
    """Sparse matrix representation for constraints."""

    rows: list[int] = Field(
        ..., description="Row indices of non-zero coefficients (0-indexed)"
    )
    cols: list[int] = Field(
        ..., description="Column indices of non-zero coefficients (0-indexed)"
    )
    values: list[float] = Field(..., description="Non-zero coefficient values")
    shape: tuple[int, int] = Field(..., description="[num_constraints, num_variables]")


class HiGHSConstraints(BaseModel):
    """Constraint specification for HiGHs."""

    # Dense format (for small problems)
    dense: list[list[float]] | None = Field(
        None, description="2D array where each row is a constraint"
    )

    # OR Sparse format (for large problems with many zeros)
    sparse: HiGHSSparseMatrix | None = Field(
        None, description="Sparse matrix representation"
    )

    sense: list[HiGHSConstraintSense] = Field(..., description="Constraint directions")
    rhs: list[float] = Field(..., description="Right-hand side values")


class HiGHSOptions(BaseModel):
    """Options for HiGHs solver."""

    # Solver Control
    time_limit: float | None = Field(None, description="Time limit in seconds")
    presolve: HiGHSPresolve | None = Field(None, description="Presolve option")
    solver: HiGHSSolver | None = Field(None, description="Solver algorithm")
    parallel: HiGHSParallel | None = Field(None, description="Parallel option")
    threads: int | None = Field(None, description="Number of threads (0=automatic)")
    random_seed: int | None = Field(None, description="Random seed for reproducibility")

    # Tolerances
    primal_feasibility_tolerance: float | None = Field(
        None, description="Default: 1e-7"
    )
    dual_feasibility_tolerance: float | None = Field(None, description="Default: 1e-7")
    ipm_optimality_tolerance: float | None = Field(None, description="Default: 1e-8")
    infinite_cost: float | None = Field(None, description="Default: 1e20")
    infinite_bound: float | None = Field(None, description="Default: 1e20")

    # Simplex Options
    simplex_strategy: int | None = Field(None, description="0-4: algorithm strategy")
    simplex_scale_strategy: int | None = Field(
        None, description="0-5: scaling strategy"
    )
    simplex_dual_edge_weight_strategy: int | None = Field(
        None, description="-1 to 2: pricing"
    )
    simplex_iteration_limit: int | None = Field(None, description="Max iterations")

    # MIP Options
    mip_detect_symmetry: bool | None = Field(None, description="Detect symmetry")
    mip_max_nodes: int | None = Field(None, description="Max branch-and-bound nodes")
    mip_rel_gap: float | None = Field(None, description="Relative gap tolerance")
    mip_abs_gap: float | None = Field(None, description="Absolute gap tolerance")
    mip_feasibility_tolerance: float | None = Field(
        None, description="MIP feasibility tolerance"
    )

    # Logging
    output_flag: bool | None = Field(None, description="Enable solver output")
    log_to_console: bool | None = Field(None, description="Console logging")
    highs_debug_level: int | None = Field(None, description="0-4: debug verbosity")

    # Algorithm-specific
    ipm_iteration_limit: int | None = Field(None, description="IPM max iterations")
    pdlp_scaling: bool | None = Field(None, description="PDLP scaling")
    pdlp_iteration_limit: int | None = Field(None, description="PDLP max iterations")

    # File I/O
    write_solution_to_file: bool | None = Field(
        None, description="Write solution to file"
    )
    solution_file: str | None = Field(None, description="Solution file path")
    write_solution_style: int | None = Field(None, description="Solution format style")


class HiGHSProblemSpec(BaseModel):
    """Problem specification for HiGHs."""

    sense: HiGHSSense = Field(..., description="Optimization sense")
    objective: HiGHSObjective = Field(..., description="Objective function")
    variables: list[HiGHSVariable] = Field(..., description="Variable specifications")
    constraints: HiGHSConstraints = Field(..., description="Constraint specifications")


class HiGHSProblem(BaseModel):
    """Complete HiGHS optimization problem."""

    problem: HiGHSProblemSpec = Field(..., description="Problem specification")
    options: HiGHSOptions | None = Field(None, description="Solver options")


class HiGHSOutput(BaseModel):
    """Standard output schema for HiGHs solver results."""

    status: HiGHSStatus = Field(..., description="Solver status")
    objective_value: float = Field(..., description="Optimal objective value")
    solution: list[float] = Field(..., description="Solution values for each variable")
    dual_solution: list[float] = Field(..., description="Dual values for constraints")
    variable_duals: list[float] = Field(..., description="Reduced costs for variables")


class HiGHSSolution(BaseModel):
    """Solution to a HiGHs problem."""

    values: dict[str, float] = Field(..., description="Variable values")
    objective_value: float | None = Field(None, description="Optimal objective value")
    status: HiGHSStatus = Field(..., description="Solver status")
    is_optimal: bool = Field(..., description="Whether solution is optimal")
    solve_time: float | None = Field(None, description="Solve time in seconds")
    iterations: int | None = Field(None, description="Number of iterations")
    dual_values: list[float] | None = Field(
        None, description="Dual values for constraints"
    )
    reduced_costs: list[float] | None = Field(
        None, description="Reduced costs for variables"
    )

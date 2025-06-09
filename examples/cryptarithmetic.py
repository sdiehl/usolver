"""
Cryptarithmetic Puzzle Solver

This module demonstrates how to solve cryptarithmetic puzzles using constraint programming.
Cryptarithmetic puzzles are mathematical word problems where letters represent digits,
and the goal is to find digit assignments that satisfy arithmetic equations.

The classic example is: SEND + MORE = MONEY
Where each letter represents a unique digit (0-9), and leading letters cannot be zero.

This implementation uses Z3 SMT solver to handle the logical constraints and arithmetic
relationships between the letter-digit mappings.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def solve_send_more_money() -> dict[str, int] | None:
    """
    Solve the classic SEND + MORE = MONEY cryptarithmetic puzzle.

    Constraints:
    - Each letter represents a unique digit (0-9)
    - Leading letters (S, M) cannot be zero
    - SEND + MORE = MONEY must hold arithmetically

    Returns:
        Dictionary mapping letters to digits if solution exists, None otherwise
    """
    try:
        from returns.result import Success

        from usolver_mcp.models.z3_models import (
            Z3Constraint,
            Z3Problem,
            Z3Variable,
            Z3VariableType,
        )
        from usolver_mcp.solvers.z3_solver import solve_problem

        # Define variables for each unique letter
        letters = ["S", "E", "N", "D", "M", "O", "R", "Y"]
        variables = [
            Z3Variable(name=letter, type=Z3VariableType.INTEGER) for letter in letters
        ]

        constraints = []

        # Each letter is a digit (0-9)
        for letter in letters:
            constraints.append(
                Z3Constraint(expression=f"And({letter} >= 0, {letter} <= 9)")
            )

        # All letters represent different digits
        different_constraints = []
        for i, letter1 in enumerate(letters):
            for letter2 in letters[i + 1 :]:
                different_constraints.append(f"{letter1} != {letter2}")

        if different_constraints:
            constraints.append(
                Z3Constraint(expression=f"And({', '.join(different_constraints)})")
            )

        # Leading letters cannot be zero
        constraints.append(Z3Constraint(expression="And(S != 0, M != 0)"))

        # Main arithmetic constraint: SEND + MORE = MONEY
        # Convert words to numbers: SEND = 1000*S + 100*E + 10*N + D
        send_expr = "1000*S + 100*E + 10*N + D"
        more_expr = "1000*M + 100*O + 10*R + E"
        money_expr = "10000*M + 1000*O + 100*N + 10*E + Y"

        constraints.append(
            Z3Constraint(expression=f"({send_expr}) + ({more_expr}) == ({money_expr})")
        )

        logger.info("Solving SEND + MORE = MONEY puzzle...")

        # Create problem
        problem = Z3Problem(
            variables=variables,
            constraints=constraints,
            description="SEND + MORE = MONEY cryptarithmetic puzzle",
        )

        # Solve the constraint system
        result = solve_problem(problem)

        # Parse the result
        if isinstance(result, Success):
            solution_obj = result.unwrap()
            if solution_obj.is_satisfiable and solution_obj.values:
                return solution_obj.values

        logger.warning("No solution found for the puzzle")
        return None

    except Exception as e:
        logger.error(f"Error solving cryptarithmetic puzzle: {e}")
        return None


def solve_general_cryptarithmetic(
    words: list[str], result_word: str
) -> dict[str, int] | None:
    """
    Solve a general cryptarithmetic puzzle of the form: word1 + word2 + ... = result_word

    Args:
        words: List of addend words
        result_word: Sum result word

    Returns:
        Dictionary mapping letters to digits if solution exists, None otherwise
    """
    try:
        from returns.result import Success

        from usolver_mcp.models.z3_models import (
            Z3Constraint,
            Z3Problem,
            Z3Variable,
            Z3VariableType,
        )
        from usolver_mcp.solvers.z3_solver import solve_problem

        # Find all unique letters
        all_letters: set[str] = set()
        for word in [*words, result_word]:
            all_letters.update(word.upper())

        letters = sorted(list(all_letters))
        variables = [
            Z3Variable(name=letter, type=Z3VariableType.INTEGER) for letter in letters
        ]

        constraints = []

        # Each letter is a digit (0-9)
        for letter in letters:
            constraints.append(
                Z3Constraint(expression=f"And({letter} >= 0, {letter} <= 9)")
            )

        # All letters represent different digits
        different_constraints = []
        for i, letter1 in enumerate(letters):
            for letter2 in letters[i + 1 :]:
                different_constraints.append(f"{letter1} != {letter2}")

        if different_constraints:
            constraints.append(
                Z3Constraint(expression=f"And({', '.join(different_constraints)})")
            )

        # Leading letters cannot be zero
        leading_letters: set[str] = set()
        for word in [*words, result_word]:
            if word:
                leading_letters.add(word[0].upper())

        leading_non_zero = []
        for letter in leading_letters:
            leading_non_zero.append(f"{letter} != 0")

        if leading_non_zero:
            constraints.append(
                Z3Constraint(expression=f"And({', '.join(leading_non_zero)})")
            )

        # Build arithmetic constraint
        def word_to_expression(word: str) -> str:
            word = word.upper()
            expr_parts = []
            for i, letter in enumerate(word):
                power = len(word) - i - 1
                if power == 0:
                    expr_parts.append(letter)
                else:
                    expr_parts.append(f"{10**power}*{letter}")
            return " + ".join(expr_parts)

        # Sum of all words equals result
        left_side = " + ".join(f"({word_to_expression(word)})" for word in words)
        right_side = word_to_expression(result_word)
        constraints.append(Z3Constraint(expression=f"({left_side}) == ({right_side})"))

        logger.info(f"Solving {' + '.join(words)} = {result_word} puzzle...")

        # Create problem
        problem = Z3Problem(
            variables=variables,
            constraints=constraints,
            description=f"Cryptarithmetic: {' + '.join(words)} = {result_word}",
        )

        # Solve the constraint system
        result = solve_problem(problem)

        # Parse the result
        if isinstance(result, Success):
            solution_obj = result.unwrap()
            if solution_obj.is_satisfiable and solution_obj.values:
                return solution_obj.values

        return None

    except Exception as e:
        logger.error(f"Error solving general cryptarithmetic puzzle: {e}")
        return None


def validate_solution(
    words: list[str], result_word: str, solution: dict[str, int]
) -> bool:
    """
    Validate that a solution satisfies the cryptarithmetic constraints.

    Args:
        words: List of addend words
        result_word: Sum result word
        solution: Dictionary mapping letters to digits

    Returns:
        True if solution is valid, False otherwise
    """
    try:

        def word_to_number(word: str) -> int:
            number = 0
            for letter in word.upper():
                number = number * 10 + solution[letter]
            return number

        # Check that all digits are unique
        if len(set(solution.values())) != len(solution):
            return False

        # Check that all digits are in range 0-9
        if not all(0 <= digit <= 9 for digit in solution.values()):
            return False

        # Check that leading letters are not zero
        for word in [*words, result_word]:
            if word and solution[word[0].upper()] == 0:
                return False

        # Check arithmetic
        word_values = [word_to_number(word) for word in words]
        result_value = word_to_number(result_word)

        return sum(word_values) == result_value

    except (KeyError, ValueError):
        return False


def print_solution_analysis(
    words: list[str], result_word: str, solution: dict[str, int]
) -> None:
    """Print detailed analysis of the cryptarithmetic solution."""
    print("\n" + "=" * 60)
    print("CRYPTARITHMETIC PUZZLE SOLUTION")
    print("=" * 60)

    print("\nLetter-to-Digit Mapping:")
    for letter in sorted(solution.keys()):
        print(f"  {letter} = {solution[letter]}")

    print(f"\nPuzzle: {' + '.join(words)} = {result_word}")
    print("\nArithmetic Verification:")

    def word_to_number(word: str) -> int:
        number = 0
        for letter in word.upper():
            number = number * 10 + solution[letter]
        return number

    word_values = []
    for word in words:
        value = word_to_number(word)
        word_values.append(value)
        print(f"  {word.upper()} = {value}")

    result_value = word_to_number(result_word)
    print(f"  {result_word.upper()} = {result_value}")

    total = sum(word_values)
    print(f"\nVerification: {' + '.join(map(str, word_values))} = {total}")
    print(f"Expected: {result_value}")
    print(f"Match: {'✓' if total == result_value else '✗'}")

    if validate_solution(words, result_word, solution):
        print("\n✓ Solution is VALID!")
    else:
        print("\n✗ Solution is INVALID!")


def main() -> None:
    """Main function to demonstrate cryptarithmetic puzzle solving."""
    print("USolver Cryptarithmetic Puzzle Solver")
    print("=====================================")

    # Solve the classic SEND + MORE = MONEY puzzle
    print("\n1. Classic SEND + MORE = MONEY Puzzle")
    solution = solve_send_more_money()

    if solution:
        print_solution_analysis(["SEND", "MORE"], "MONEY", solution)
    else:
        print("No solution found for SEND + MORE = MONEY")

    # Solve another puzzle: TWO + TWO = FOUR
    print("\n" + "=" * 60)
    print("\n2. TWO + TWO = FOUR Puzzle")
    solution2 = solve_general_cryptarithmetic(["TWO", "TWO"], "FOUR")

    if solution2:
        print_solution_analysis(["TWO", "TWO"], "FOUR", solution2)
    else:
        print("No solution found for TWO + TWO = FOUR")

    # Solve: ONE + ONE = TWO
    print("\n" + "=" * 60)
    print("\n3. ONE + ONE = TWO Puzzle")
    solution3 = solve_general_cryptarithmetic(["ONE", "ONE"], "TWO")

    if solution3:
        print_solution_analysis(["ONE", "ONE"], "TWO", solution3)
    else:
        print("No solution found for ONE + ONE = TWO")


def test_cryptarithmetic_solver() -> None:
    """Test function for pytest compatibility."""
    # Test SEND + MORE = MONEY
    solution1 = solve_send_more_money()
    assert solution1 is not None, "Should find solution for SEND + MORE = MONEY"
    assert validate_solution(
        ["SEND", "MORE"], "MONEY", solution1
    ), "Solution should be valid"

    # Test TWO + TWO = FOUR
    solution2 = solve_general_cryptarithmetic(["TWO", "TWO"], "FOUR")
    if solution2:  # This puzzle may not have a solution
        assert validate_solution(
            ["TWO", "TWO"], "FOUR", solution2
        ), "Solution should be valid"

    # Test ONE + ONE = TWO
    solution3 = solve_general_cryptarithmetic(["ONE", "ONE"], "TWO")
    if solution3:
        assert validate_solution(
            ["ONE", "ONE"], "TWO", solution3
        ), "Solution should be valid"

    print("All cryptarithmetic tests passed!")


if __name__ == "__main__":
    main()

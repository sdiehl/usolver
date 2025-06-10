import os
import sys
from pathlib import Path

from anthropic import Anthropic

with open(Path(__file__).parent / "prompt.md") as f:
    prompt = f.read()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: Set ANTHROPIC_API_KEY environment variable")
    sys.exit(1)

anthropic = Anthropic()
response = anthropic.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    messages=[{"role": "user", "content": prompt}],
    mcp_servers=[
        {
            "type": "url",
            "url": "http://localhost:8081",
            "name": "usolver",
            "tool_configuration": {"enabled": True},
            "allowed_tools": [
                "solve_z3",
                "solve_z3_simple",
                "solve_highs_problem",
                "simple_highs_solver",
                "solve_cvxpy_problem",
                "simple_cvxpy_solver",
                "solve_ortools_problem",
            ],
        }
    ],
    extra_headers={"anthropic-beta": "mcp-client-2025-04-04"},
)

for block in response.content:
    if hasattr(block, "text"):
        print(block.text)

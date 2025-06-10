import os
import sys
from pathlib import Path

import anthropic

with open(Path(__file__).parent / "prompt.md") as f:
    prompt = f.read()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: Set ANTHROPIC_API_KEY environment variable")
    sys.exit(1)

client = anthropic.Anthropic()

flowsheet_file = client.beta.files.upload(
    file=(
        "flowsheet_data.csv",
        open(Path(__file__).parent / "flowsheet_data.xlsx", "rb"),
        "text/csv",
    )
)

metadata_file = client.beta.files.upload(
    file=(
        "flowsheet_metadata.csv",
        open(Path(__file__).parent / "flowsheet_metadata.xlsx", "rb"),
        "text/csv",
    )
)

response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4000,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": flowsheet_file.id},
                },
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": metadata_file.id},
                },
            ],
        }
    ],
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
    betas=["files-api-2025-04-14"],
    extra_headers={"anthropic-beta": "mcp-client-2025-04-04"},
)

for block in response.content:
    if hasattr(block, "text"):
        print(block.text)

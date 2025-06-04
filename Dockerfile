FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml .
COPY uv.lock .
COPY usolver_mcp/ ./usolver_mcp/
COPY examples/ ./examples/

# Expose the default MCP port
EXPOSE 8081

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8081/healthcheck || exit 1

# Run the server with SSE transport
CMD ["uv", "run", "mcp", "run", "usolver_mcp/server/main.py","--transport", "sse"]
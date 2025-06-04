#!/usr/bin/env python3
"""
MCP Server Installer Script
Installs the local MCP server to Claude Desktop and Cursor configurations
"""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

SERVER_NAME = "usolver"


def get_config_paths() -> tuple[Path, Path]:
    """Get the config file paths for Claude Desktop and Cursor"""
    system = platform.system()

    if system == "Darwin":  # macOS
        claude_config = (
            Path.home()
            / "Library/Application Support/Claude/claude_desktop_config.json"
        )
        cursor_config = Path.home() / ".cursor/mcp.json"
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        claude_config = Path(appdata) / "Claude/claude_desktop_config.json"
        cursor_config = Path.home() / ".cursor/mcp.json"
    else:  # Linux and others
        claude_config = Path.home() / ".config/Claude/claude_desktop_config.json"
        cursor_config = Path.home() / ".cursor/mcp.json"

    return claude_config, cursor_config


def load_or_create_config(config_path: Path) -> dict[str, Any]:
    """Load existing config or create a new one"""
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"Warning: Could not read {config_path}, creating new config")

    return {"mcpServers": {}}


def save_config(config_path: Path, config_data: dict[str, Any]) -> None:
    """Save config to file, creating directories if needed"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)


def get_uv_command() -> str:
    """Get the uv command path"""
    possible_paths = [
        "/opt/homebrew/bin/uv",  # Homebrew on macOS
        "/usr/local/bin/uv",  # Manual install
        "uv",  # In PATH
    ]

    for path in possible_paths:
        if Path(path).exists() or path == "uv":
            return path

    return "uv"  # Fallback


def install_to_config(config_path: Path, script_dir: Path, server_name: str) -> bool:
    """Install MCP server configuration to a config file"""
    config = load_or_create_config(config_path)

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Build server configuration with all required dependencies
    server_config = {
        "command": get_uv_command(),
        "args": [
            "run",
            "--directory",
            str(script_dir),
            "usolver_mcp/server/main.py",
        ],
    }

    # Add our server configuration
    config["mcpServers"][server_name] = server_config

    save_config(config_path, config)
    return True


def main() -> None:
    """Install MCP server to local configurations"""
    script_dir = Path(__file__).parent.absolute()
    server_name = SERVER_NAME

    print("MCP Server Installer")
    print("======================")
    print()
    print("This will install the Z3 MCP solver to your local client configurations.")
    print(f"Installation directory: {script_dir}")
    print(f"Server name: {server_name}")
    print()

    # Ask for confirmation
    while True:
        response = (
            input("Do you want to proceed with the installation? (y/n): ")
            .lower()
            .strip()
        )
        if response in ["y", "yes"]:
            break
        elif response in ["n", "no"]:
            print("Installation cancelled.")
            sys.exit(0)
        else:
            print("Please enter 'y' for yes or 'n' for no.")

    print()

    # Get config paths
    claude_config, cursor_config = get_config_paths()
    installed_to = []

    # Install to Claude Desktop
    try:
        install_to_config(claude_config, script_dir, server_name)
        print(f"✓ Installed to Claude Desktop: {claude_config}")
        installed_to.append("Claude Desktop")
    except Exception as e:
        print(f"✗ Failed to install to Claude Desktop: {e}")

    # Install to Cursor (if config directory exists)
    if cursor_config.parent.exists():
        try:
            install_to_config(cursor_config, script_dir, server_name)
            print(f"✓ Installed to Cursor: {cursor_config}")
            installed_to.append("Cursor")
        except Exception as e:
            print(f"✗ Failed to install to Cursor: {e}")
    else:
        print("• Cursor config directory not found, skipping")

    print()
    if installed_to:
        print("Installation completed successfully!")
        print(f"Installed to: {', '.join(installed_to)}")
        print()
        print("Please restart your client(s) to use the MCP server.")
    else:
        print("Installation failed - no configurations were updated.")
        sys.exit(1)


if __name__ == "__main__":
    main()

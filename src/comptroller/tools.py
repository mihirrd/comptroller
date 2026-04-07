import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read file contents and return with line count header."""
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        content = file_path.read_text()
        line_count = len(content.splitlines())
        return f"[{line_count} lines]\n{content}"
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to file and return confirmation."""
    try:
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        size = len(content)
        return f"Successfully wrote {size} bytes to {path}"
    except Exception as e:
        return f"Error writing file {path}: {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """List files in directory with sizes."""
    try:
        dir_path = Path(path).expanduser().resolve()
        if not dir_path.exists():
            return f"Error: Directory not found: {path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for item in sorted(dir_path.iterdir()):
            if item.is_file():
                size = item.stat().st_size
                entries.append(f"  {item.name} ({size} bytes)")
            elif item.is_dir():
                entries.append(f"  {item.name}/ (directory)")

        if not entries:
            return f"Directory {path} is empty"
        return f"Contents of {path}:\n" + "\n".join(entries)
    except Exception as e:
        return f"Error listing directory {path}: {str(e)}"


@tool
def run_shell(command: str) -> str:
    """Execute shell command with 30s timeout, return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output.append(f"Exit code: {result.returncode}")

        return "\n".join(output) if output else "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


# Export tools list and dict
TOOLS = [read_file, write_file, list_directory, run_shell]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

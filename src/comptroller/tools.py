import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# When set (e.g. by main.run / run_graph), file tools reject paths outside this directory.
WORKSPACE_ROOT_ENV = "COMPTROLLER_WORKSPACE_ROOT"

_DEFAULT_GREP_MAX_MATCHES = 100
_DEFAULT_GLOB_MAX_RESULTS = 500
_RG_TIMEOUT_SEC = 60
_PATCH_APPLY_TIMEOUT_SEC = 120

# If grep returns no lines and the pattern looks like a literal path segment (not regex), nudge the model.
_GREP_FILENAME_LIKE = re.compile(r"^[\w.-]+$")


def _recursive_glob_pattern(pattern: str) -> str:
    """Make single-component patterns match any depth under root (Path.glob is non-recursive by default)."""
    p = pattern.strip()
    if not p or "**" in p or "/" in p:
        return p
    return f"**/{p}"


def _grep_filename_like_hint(pattern: str) -> str:
    if not _GREP_FILENAME_LIKE.fullmatch(pattern):
        return ""
    return (
        " Hint: `grep` searches file *contents*, not paths. "
        "To find files by name, use `glob_files` with the same pattern (recursive under `root_directory`)."
    )


def set_workspace_root_for_tools(workspace_root: str | None) -> None:
    """Bind file tools to a workspace root (SWE-bench / repo runs). Unset when None."""
    if workspace_root:
        os.environ[WORKSPACE_ROOT_ENV] = str(Path(workspace_root).expanduser().resolve())
    else:
        os.environ.pop(WORKSPACE_ROOT_ENV, None)


def _configured_workspace_root() -> Path | None:
    raw = os.environ.get(WORKSPACE_ROOT_ENV)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _resolve_path(path: str) -> tuple[Path | None, str | None]:
    """Resolve path and enforce workspace jail when WORKSPACE_ROOT_ENV is set.

    Relative paths are resolved against the configured workspace root, not
    ``os.getcwd()``, so tools work when the process cwd differs from the repo
    (e.g. SWE-bench harness run from ``harness/`` while ``workspace_root`` is a temp clone).
    """
    root = _configured_workspace_root()
    try:
        raw = Path(path).expanduser()
        if root is not None and not raw.is_absolute():
            p = (root / raw).resolve()
        else:
            p = raw.resolve()
    except Exception as e:
        return None, f"Error resolving path {path!r}: {e}"
    if root is not None:
        try:
            p.relative_to(root)
        except ValueError:
            return None, f"Error: path escapes workspace_root ({root}): {p}"
    return p, None


def _read_file_slice(
    content: str,
    *,
    start_line: Optional[int],
    end_line: Optional[int],
) -> tuple[str, int, int, int]:
    """Return (body, total_lines, first_line_no, last_line_no) for a 1-based inclusive range.

    If both ``start_line`` and ``end_line`` are None, the full file is returned and
    ``first_line_no``/``last_line_no`` are 1 and total_lines (or 0,0 for empty).
    """
    lines = content.splitlines()
    total = len(lines)
    if start_line is None and end_line is None:
        body = content
        if total == 0:
            return body, 0, 0, 0
        return body, total, 1, total

    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1 when provided")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1 when provided")

    start = 1 if start_line is None else start_line
    end = total if end_line is None else end_line
    if start > end:
        start, end = end, start
    if total == 0:
        return "", 0, start, end
    # Explicit start past EOF → empty slice (do not clamp to last line).
    if start_line is not None and start > total:
        return "", total, start, end
    start_eff = max(1, start)
    end_eff = min(end, total)
    if start_eff > end_eff:
        return "", total, start_eff, end_eff
    chunk = lines[start_eff - 1 : end_eff]
    body = "\n".join(chunk)
    return body, total, start_eff, end_eff


@tool
def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read file contents with an optional 1-based inclusive line range.

    Omit ``start_line`` and ``end_line`` to return the full file. Pass either or both
    to return only those lines (end is clamped to file length). Line numbers match
    typical editor / ``grep -n`` output (first line is 1).
    """
    file_path, err = _resolve_path(path)
    if err:
        return err
    try:
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        content = file_path.read_text()
        try:
            body, total, lo, hi = _read_file_slice(
                content, start_line=start_line, end_line=end_line
            )
        except ValueError as e:
            return f"Error: {e}"

        if start_line is None and end_line is None:
            return f"[{total} lines]\n{body}"

        if total == 0:
            return f"[lines {lo}-{hi} of 0 total]\n"
        if not body and lo > total:
            return (
                f"[lines {lo}-{hi} requested; file has {total} lines]\n"
                "(no lines in range)"
            )
        return f"[lines {lo}-{hi} of {total} total]\n{body}"
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """Write full file content and return confirmation.

    Prefer this over many search_replace steps when the file is small or you are rewriting several functions at once.
    """
    file_path, err = _resolve_path(path)
    if err:
        return err
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        size = len(content)
        return f"Successfully wrote {size} bytes to {path}"
    except Exception as e:
        return f"Error writing file {path}: {str(e)}"


@tool
def search_replace(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace old_string with new_string in a text file. Best for a single, unambiguous edit.

    old_string must match the file exactly (including whitespace). Include several lines of context so it appears only once.
    If you get "appears N times", widen old_string with more context instead of using replace_all unless you mean to change every copy.
    When replace_all is false (default), old_string must appear exactly once.
    When replace_all is true, every occurrence is replaced (risky for shared snippets like `self.items = ...`).
    """
    file_path, err = _resolve_path(path)
    if err:
        return err
    try:
        if not file_path.is_file():
            return f"Error: Not a file: {path}"
        content = file_path.read_text()
        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if not replace_all and count != 1:
            return f"Error: old_string appears {count} times; use replace_all=true or make old_string unique"
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
        file_path.write_text(new_content)
        return f"Successfully replaced {count} occurrence(s) in {path}"
    except Exception as e:
        return f"Error in search_replace for {path}: {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """List files in directory with sizes."""
    dir_path, err = _resolve_path(path)
    if err:
        return err
    try:
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
def glob_files(pattern: str, root_directory: str = ".") -> str:
    """List file paths matching a glob under root_directory.

    Patterns without "/" or "**" are searched recursively (e.g. "cart.py" matches subdir/cart.py).
    Use "prefix/*.py" or explicit "**/*.py" when you need a specific tree shape.
    """
    root, err = _resolve_path(root_directory)
    if err:
        return err
    try:
        if not root.is_dir():
            return f"Error: root_directory is not a directory: {root_directory}"
        glob_pat = _recursive_glob_pattern(pattern)
        matches: list[Path] = []
        for p in sorted(root.glob(glob_pat)):
            if p.is_file():
                matches.append(p)
            if len(matches) >= _DEFAULT_GLOB_MAX_RESULTS:
                break

        if not matches:
            return (
                f"No files matched pattern {pattern!r} under {root_directory}"
                + (f" (resolved as {glob_pat!r})" if glob_pat != pattern.strip() else "")
            )
        lines = [str(p) for p in matches]
        msg = "\n".join(lines)
        if len(matches) >= _DEFAULT_GLOB_MAX_RESULTS:
            msg += f"\n... (stopped at {_DEFAULT_GLOB_MAX_RESULTS} paths)"
        return msg
    except Exception as e:
        return f"Error glob_files: {str(e)}"


def _validate_unified_diff_paths(patch_text: str, root: Path) -> str | None:
    """Ensure +++ paths stay inside root (after stripping a/ b/ prefixes)."""
    root = root.resolve()
    for line in patch_text.splitlines():
        if not line.startswith("+++ "):
            continue
        rest = line[4:].strip()
        if "dev/null" in rest:
            continue
        path_part = rest.split("\t", 1)[0].strip()
        for prefix in ("b/", "a/"):
            if path_part.startswith(prefix):
                path_part = path_part[len(prefix) :]
                break
        if not path_part or path_part == "/dev/null":
            continue
        if os.path.isabs(path_part):
            candidate = Path(path_part).resolve()
        else:
            candidate = (root / path_part).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return f"Error: patch references path outside workspace: {candidate}"
    return None


def _grep_is_gnu_grep(grep_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [grep_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "GNU" in (proc.stdout or "")


def _format_grep_stdout_lines(lines: list[str], max_matches: int) -> str:
    """Normalize path:line:rest output from rg or grep (first colon may be in Windows path — best effort)."""
    if not lines:
        return ""
    trimmed = lines[:max_matches]
    wr = _configured_workspace_root()
    out_lines: list[str] = []
    for line in trimmed:
        if ":" not in line:
            out_lines.append(line[:600])
            continue
        first_colon = line.index(":")
        if line.find(":", first_colon + 1) == -1:
            out_lines.append(line[:600])
            continue
        file_part = line[:first_colon]
        rest = line[first_colon + 1 :]
        try:
            fp = Path(file_part).resolve()
            if wr is not None:
                try:
                    file_part = str(fp.relative_to(wr))
                except ValueError:
                    file_part = str(fp)
            else:
                file_part = str(fp)
        except Exception:
            pass
        out_lines.append(f"{file_part}:{rest}"[:600])
    msg = "\n".join(out_lines)
    if len(lines) > max_matches:
        msg += f"\n... (stopped at {max_matches} matches)"
    return msg


def _run_ripgrep(
    pattern: str,
    target: Path,
    max_matches: int,
    glob_filter: str,
) -> str:
    rg = shutil.which("rg")
    assert rg is not None
    cmd: list[str] = [
        rg,
        "-n",
        "--color",
        "never",
        "--max-columns",
        "512",
        "--regexp",
        pattern,
        str(target),
    ]
    if glob_filter:
        cmd = cmd[:1] + ["--glob", glob_filter] + cmd[1:]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_RG_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error: ripgrep failed to run: {e}"
    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "").strip()
        if "regex parse error" in err.lower() or "error parsing regex" in err.lower():
            return f"Error: ripgrep rejected pattern (Rust regex): {err}"
        return f"Error: ripgrep failed (exit {proc.returncode}):\n{err}"
    lines = proc.stdout.splitlines()
    if not lines:
        return f"No matches for pattern {pattern!r} under {target}.{_grep_filename_like_hint(pattern)}"
    return _format_grep_stdout_lines(lines, max_matches)


def _run_system_grep(
    pattern: str,
    target: Path,
    max_matches: int,
    glob_filter: str,
) -> str:
    g = shutil.which("grep")
    assert g is not None
    gnu = _grep_is_gnu_grep(g)

    if target.is_file():
        if glob_filter and not fnmatch.fnmatch(target.name, glob_filter):
            return f"No matches (file name does not match glob_filter {glob_filter!r})"
        cmd = [g, "-n", "-E", "--", pattern, str(target)]
    else:
        if glob_filter and not gnu:
            return (
                "Error: glob_filter with a directory requires GNU grep or ripgrep (rg). "
                "Install rg, use GNU grep, search a single file, or omit glob_filter."
            )
        if target.is_dir():
            if gnu:
                cmd = [
                    g,
                    "-r",
                    "-n",
                    "-E",
                    "--binary-files=without-match",
                    "--exclude-dir=.git",
                    "--exclude-dir=node_modules",
                    "--exclude-dir=__pycache__",
                    "--exclude-dir=.venv",
                    "--exclude-dir=venv",
                ]
                if glob_filter:
                    cmd.append(f"--include={glob_filter}")
                cmd.extend(["--", pattern, str(target)])
            else:
                # BSD grep: no --include / --exclude-dir; glob_filter on dirs already rejected above
                cmd = [g, "-R", "-n", "-E", "--", pattern, str(target)]
        else:
            return f"Error: path is not a file or directory: {target}"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_RG_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"Error: grep failed to run: {e}"
    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "").strip()
        return f"Error: grep failed (exit {proc.returncode}):\n{err}"
    lines = proc.stdout.splitlines()
    if not lines:
        return f"No matches for pattern {pattern!r} under {target}.{_grep_filename_like_hint(pattern)}"
    return _format_grep_stdout_lines(lines, max_matches)


@tool
def grep(
    pattern: str,
    path: str = ".",
    max_matches: int = _DEFAULT_GREP_MAX_MATCHES,
    glob_filter: str = "",
) -> str:
    """Search file *contents* with ripgrep (rg) if installed; otherwise grep -E (extended regex).

    This is not for finding files by name; use glob_files for that. ``pattern`` is a regex (rg: Rust regex; grep: ERE).

    glob_filter: with rg, passed as --glob (e.g. "*.py"). With GNU grep and a directory, passed as --include.
    BSD grep does not support glob_filter on directories — install rg or GNU grep, or omit glob_filter.
    """
    target, err = _resolve_path(path)
    if err:
        return err

    if shutil.which("rg"):
        return _run_ripgrep(pattern, target, max_matches, glob_filter)
    if shutil.which("grep"):
        return _run_system_grep(pattern, target, max_matches, glob_filter)
    return (
        "Error: neither ripgrep (rg) nor grep found on PATH. "
        "Install ripgrep: https://github.com/BurntSushi/ripgrep"
    )


@tool
def apply_patch(patch_text: str) -> str:
    """Apply a unified diff (git-style) under the workspace root. Prefers `git apply`; uses `patch -p1` if no .git.

    Patch paths must stay inside workspace_root when COMPTROLLER_WORKSPACE_ROOT is set.
    """
    if not patch_text.strip():
        return "Error: empty patch"
    root = _configured_workspace_root()
    if root is None:
        root = Path.cwd().resolve()
    else:
        root = root.resolve()

    verr = _validate_unified_diff_paths(patch_text, root)
    if verr:
        return verr

    if (root / ".git").exists():
        check = subprocess.run(
            ["git", "apply", "--check"],
            cwd=root,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=_PATCH_APPLY_TIMEOUT_SEC,
        )
        if check.returncode != 0:
            detail = (check.stderr or check.stdout or "").strip()
            return f"Error: git apply --check failed:\n{detail}"
        applied = subprocess.run(
            ["git", "apply"],
            cwd=root,
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=_PATCH_APPLY_TIMEOUT_SEC,
        )
        if applied.returncode != 0:
            detail = (applied.stderr or applied.stdout or "").strip()
            return f"Error: git apply failed:\n{detail}"
        return "Successfully applied patch (git apply)."

    patch_bin = shutil.which("patch")
    if not patch_bin:
        return (
            "Error: no .git directory and `patch` not found on PATH. "
            "Use a git clone or install GNU patch."
        )
    dry = subprocess.run(
        [patch_bin, "-p1", "--forward", "--dry-run"],
        cwd=root,
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=_PATCH_APPLY_TIMEOUT_SEC,
    )
    if dry.returncode != 0:
        detail = (dry.stderr or dry.stdout or "").strip()
        return f"Error: patch dry-run failed:\n{detail}"
    real = subprocess.run(
        [patch_bin, "-p1", "--forward"],
        cwd=root,
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=_PATCH_APPLY_TIMEOUT_SEC,
    )
    if real.returncode != 0:
        detail = (real.stderr or real.stdout or "").strip()
        return f"Error: patch failed:\n{detail}"
    return "Successfully applied patch (patch -p1)."


@tool
def run_shell(command: str) -> str:
    """Execute shell command with 30s timeout, return stdout + stderr."""
    try:
        cwd = _configured_workspace_root()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(cwd) if cwd else None,
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
TOOLS = [
    read_file,
    write_file,
    search_replace,
    apply_patch,
    list_directory,
    glob_files,
    grep,
    run_shell,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

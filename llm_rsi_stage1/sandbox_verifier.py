"""
sandbox_verifier.py - Deterministic Subprocess Execution Sandbox for Verifying LLM Code.

Enforces:
- AST security scanning (rejects arbitrary process spawning or system deletion)
- Strict wall-clock execution timeout (3.0 seconds) to safely catch infinite loops
- Full capture of stdout, stderr, tracebacks, and assertion failures
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional


class VerificationResult(NamedTuple):
    passed: bool
    error_type: Optional[str]  # None, "SYNTAX_ERROR", "TIMEOUT", "ASSERTION_ERROR", "RUNTIME_ERROR", "SECURITY_BLOCKED"
    message: str
    execution_time_sec: float


# Disallowed AST nodes and imports for local sandbox safety
BLOCKED_MODULES = {"os", "subprocess", "shutil", "sys", "socket", "urllib", "requests", "pty"}


def check_ast_safety(code: str) -> Optional[str]:
    """Scans code AST for syntax errors and unsafe system module imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SYNTAX_ERROR: {e.msg} at line {e.lineno}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in BLOCKED_MODULES:
                    return f"SECURITY_BLOCKED: Import of '{root_pkg}' is restricted in sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in BLOCKED_MODULES:
                    return f"SECURITY_BLOCKED: Import from '{root_pkg}' is restricted in sandbox."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "__import__"}:
                    return f"SECURITY_BLOCKED: Direct call to '{node.func.id}' is restricted."

    return None


def verify_solution(solution_code: str, unit_tests: str, timeout_sec: float = 3.0) -> VerificationResult:
    """
    Executes solution_code combined with unit_tests in an isolated subprocess.
    Returns VerificationResult indicating pass/fail status and diagnostics.
    """
    import time

    # 1. AST Safety & Syntax Check
    ast_err = check_ast_safety(solution_code)
    if ast_err:
        if ast_err.startswith("SYNTAX_ERROR"):
            return VerificationResult(passed=False, error_type="SYNTAX_ERROR", message=ast_err, execution_time_sec=0.0)
        else:
            return VerificationResult(passed=False, error_type="SECURITY_BLOCKED", message=ast_err, execution_time_sec=0.0)

    # 2. Assemble test harness
    full_script = f"""# Auto-generated verification harness
{solution_code.strip()}

# Unit Test Assertions
if __name__ == '__main__':
{chr(10).join('    ' + line for line in unit_tests.strip().splitlines() if line.strip())}
    print("VERIFICATION_PASSED_OK")
"""

    start_t = time.perf_counter()

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(full_script)

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        elapsed = round(time.perf_counter() - start_t, 3)

        if proc.returncode == 0 and "VERIFICATION_PASSED_OK" in proc.stdout:
            return VerificationResult(
                passed=True,
                error_type=None,
                message="All assertions passed successfully.",
                execution_time_sec=elapsed
            )

        # Non-zero exit code: check if AssertionError or Runtime Exception
        stderr = proc.stderr.strip()
        if "AssertionError" in stderr:
            return VerificationResult(
                passed=False,
                error_type="ASSERTION_ERROR",
                message=stderr.splitlines()[-1] if stderr else "Assertion failed.",
                execution_time_sec=elapsed
            )
        else:
            last_line = stderr.splitlines()[-1] if stderr else "Unknown runtime error."
            return VerificationResult(
                passed=False,
                error_type="RUNTIME_ERROR",
                message=last_line,
                execution_time_sec=elapsed
            )

    except subprocess.TimeoutExpired:
        elapsed = round(time.perf_counter() - start_t, 3)
        return VerificationResult(
            passed=False,
            error_type="TIMEOUT",
            message=f"Execution timed out after {timeout_sec}s (likely infinite loop).",
            execution_time_sec=elapsed
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

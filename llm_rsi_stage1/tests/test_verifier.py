"""
test_verifier.py - Unit tests for sandbox_verifier.py safety and error classification.
"""

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_rsi_stage1.sandbox_verifier import verify_solution, check_ast_safety


class TestSandboxVerifier(unittest.TestCase):

    def test_correct_solution_passes(self):
        solution = """
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
"""
        tests = """
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
"""
        res = verify_solution(solution, tests)
        self.assertTrue(res.passed)
        self.assertIsNone(res.error_type)
        self.assertIn("All assertions passed", res.message)

    def test_assertion_failure_caught(self):
        solution = """
def is_palindrome(s: str) -> bool:
    return False # deliberately wrong
"""
        tests = """
assert is_palindrome("") == True
"""
        res = verify_solution(solution, tests)
        self.assertFalse(res.passed)
        self.assertEqual(res.error_type, "ASSERTION_ERROR")

    def test_syntax_error_caught(self):
        solution = """
def broken_syntax(s
    return 123
"""
        tests = "assert True"
        res = verify_solution(solution, tests)
        self.assertFalse(res.passed)
        self.assertEqual(res.error_type, "SYNTAX_ERROR")

    def test_infinite_loop_timeout_caught(self):
        solution = """
def infinite_loop():
    while True:
        pass
"""
        tests = "infinite_loop()"
        res = verify_solution(solution, tests, timeout_sec=1.0)
        self.assertFalse(res.passed)
        self.assertEqual(res.error_type, "TIMEOUT")
        self.assertIn("timed out", res.message)

    def test_blocked_modules_rejected(self):
        solution = """
import os
def bad_func():
    os.system("echo hello")
"""
        tests = "assert True"
        res = verify_solution(solution, tests)
        self.assertFalse(res.passed)
        self.assertEqual(res.error_type, "SECURITY_BLOCKED")


if __name__ == "__main__":
    unittest.main()

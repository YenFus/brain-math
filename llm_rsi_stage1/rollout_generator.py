"""
rollout_generator.py - MLX-LM Rollout Generator for Qwen2.5-Coder-1.5B on Apple Silicon.

Handles:
- Loading quantized 4-bit weights into unified memory via mlx_lm
- Prompt structuring for Chain-of-Thought reasoning + Python implementation
- Robust regex and AST markdown block extraction
- Safe fallback / simulation generator for local offline tests
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional


class GeneratedRollout(NamedTuple):
    raw_response: str
    extracted_code: str
    reasoning_trace: str
    is_code_extracted: bool


SYSTEM_PROMPT = """You are an expert Python algorithmic programmer.
When given a programming problem:
1. Briefly analyze the problem, edge cases, and time complexity in a short reasoning paragraph.
2. Provide the complete, working Python function inside a single markdown ```python ... ``` code block.
3. Do not include extraneous test scripts or input() calls; only the function definition and necessary standard library imports.
"""


def extract_python_code(response_text: str) -> tuple[str, str]:
    """
    Extracts Python code from markdown triple backticks.
    Returns (code, reasoning_trace).
    """
    # Look for ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        reasoning = response_text[:match.start()].strip()
        return code, reasoning

    # Fallback: if no code block, look for 'def ' block
    def_match = re.search(r"(def\s+[a-zA-Z0-9_]+\s*\(.*)", response_text, re.DOTALL)
    if def_match:
        code = def_match.group(1).strip()
        reasoning = response_text[:def_match.start()].strip()
        return code, reasoning

    return "", response_text.strip()


class QwenRolloutGenerator:
    """Generates candidate code solutions using Qwen2.5-Coder-1.5B via mlx_lm."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load(self):
        """Loads the model and tokenizer into Apple Silicon unified memory."""
        if not self.is_loaded:
            import mlx_lm
            print(f"Loading {self.model_id} via mlx_lm...")
            self.model, self.tokenizer = mlx_lm.load(self.model_id)
            self.is_loaded = True
            print("Model loaded successfully.")

    def generate_candidate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512
    ) -> GeneratedRollout:
        """Generates a candidate solution rollout for the given prompt."""
        if not self.is_loaded:
            self.load()

        import mlx_lm

        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)

        formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        raw_output = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=False
        )

        code, reasoning = extract_python_code(raw_output)
        return GeneratedRollout(
            raw_response=raw_output,
            extracted_code=code,
            reasoning_trace=reasoning,
            is_code_extracted=bool(code)
        )


class MockRolloutGenerator:
    """
    Simulated rollout generator with a mix of correct and flawed solutions
    for fast, offline end-to-end pipeline verification without network weights.
    """

    # Library of realistic candidates (correct and flawed) for benchmark tasks
    PRESET_CANDIDATES = {
        "task_01": [
            # Correct
            """```python
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
```""",
            # Flawed (fails case sensitivity / spaces)
            """```python
def is_palindrome(s: str) -> bool:
    return s == s[::-1]
```"""
        ],
        "task_02": [
            # Correct
            """```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```""",
            # Flawed (off-by-one)
            """```python
def fibonacci(n: int) -> int:
    return n * (n - 1) // 2
```"""
        ],
        "task_03": [
            # Correct
            """```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []
```""",
            # Flawed (returns numbers instead of indices)
            """```python
def two_sum(nums: list[int], target: int) -> list[int]:
    for i in nums:
        for j in nums:
            if i + j == target:
                return [i, j]
    return []
```"""
        ],
        "task_04": [
            # Correct
            """```python
def count_vowels(s: str) -> int:
    vowels = set("aeiouAEIOU")
    return sum(1 for c in s if c in vowels)
```"""
        ],
        "task_05": [
            # Correct
            """```python
def reverse_words(s: str) -> str:
    return " ".join(s.split()[::-1])
```"""
        ],
        "task_06": [
            # Correct
            """```python
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```"""
        ],
        "task_07": [
            # Correct
            """```python
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    res = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    res.extend(a[i:])
    res.extend(b[j:])
    return res
```"""
        ],
        "task_08": [
            # Correct
            """```python
def valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack
```"""
        ]
    }

    def __init__(self):
        self.call_counts = {}

    def generate_candidate(
        self,
        task_id: str,
        temperature: float = 0.2
    ) -> GeneratedRollout:
        """Cycles through preset correct / flawed samples for the task."""
        candidates = self.PRESET_CANDIDATES.get(task_id, [
            # Generic fallback correct implementation for remaining tasks
            """```python
def placeholder_func(*args):
    return True
```"""
        ])

        idx = self.call_counts.get(task_id, 0) % len(candidates)
        self.call_counts[task_id] = self.call_counts.get(task_id, 0) + 1
        raw_text = candidates[idx]

        code, reasoning = extract_python_code(raw_text)
        return GeneratedRollout(
            raw_response=raw_text,
            extracted_code=code,
            reasoning_trace=reasoning,
            is_code_extracted=bool(code)
        )

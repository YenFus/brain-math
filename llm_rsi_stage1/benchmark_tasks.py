"""
benchmark_tasks.py - 15 Algorithmic Coding Tasks for Stage 1 Self-Improvement Evaluation.

Each task has an exact prompt, expected function name, and strict deterministic unit test suite.
"""

from typing import Dict, List, NamedTuple

class CodingTask(NamedTuple):
    task_id: str
    name: str
    prompt: str
    function_name: str
    unit_tests: str

BENCHMARK_TASKS: List[CodingTask] = [
    CodingTask(
        task_id="task_01",
        name="Valid Palindrome",
        prompt="Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome, considering only alphanumeric characters and ignoring cases. Otherwise, return False.",
        function_name="is_palindrome",
        unit_tests="""
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
assert is_palindrome("0P") == False
assert is_palindrome("Was it a car or a cat I saw?") == True
"""
    ),
    CodingTask(
        task_id="task_02",
        name="N-th Fibonacci Number",
        prompt="Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci number where F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2). Assume n >= 0.",
        function_name="fibonacci",
        unit_tests="""
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
assert fibonacci(15) == 610
"""
    ),
    CodingTask(
        task_id="task_03",
        name="Two Sum",
        prompt="Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the 0-indexed indices of the two numbers in `nums` that add up to `target`. Exactly one solution exists.",
        function_name="two_sum",
        unit_tests="""
assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
assert sorted(two_sum([3, 3], 6)) == [0, 1]
assert sorted(two_sum([1, 5, 8, 3], 11)) == [2, 3]
"""
    ),
    CodingTask(
        task_id="task_04",
        name="Count Vowels",
        prompt="Write a Python function `count_vowels(s: str) -> int` that returns the total count of vowels (a, e, i, o, u, case-insensitive) in the input string.",
        function_name="count_vowels",
        unit_tests="""
assert count_vowels("hello world") == 3
assert count_vowels("PYTHON") == 1
assert count_vowels("rhythm") == 0
assert count_vowels("AEIOUaeiou") == 10
"""
    ),
    CodingTask(
        task_id="task_05",
        name="Reverse Words",
        prompt="Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a sentence. Strip extra spaces so that words are separated by a single space with no leading or trailing spaces.",
        function_name="reverse_words",
        unit_tests="""
assert reverse_words("the sky is blue") == "blue is sky the"
assert reverse_words("  hello world  ") == "world hello"
assert reverse_words("a good   example") == "example good a"
"""
    ),
    CodingTask(
        task_id="task_06",
        name="Primality Test",
        prompt="Write a Python function `is_prime(n: int) -> bool` that returns True if the integer n is a prime number, and False otherwise.",
        function_name="is_prime",
        unit_tests="""
assert is_prime(1) == False
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(100) == False
assert is_prime(97) == True
"""
    ),
    CodingTask(
        task_id="task_07",
        name="Merge Sorted Lists",
        prompt="Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` that merges two already-sorted integer lists into a single sorted list.",
        function_name="merge_sorted",
        unit_tests="""
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([], [1, 2]) == [1, 2]
assert merge_sorted([5], []) == [5]
assert merge_sorted([1, 1], [1, 1]) == [1, 1, 1, 1]
"""
    ),
    CodingTask(
        task_id="task_08",
        name="Valid Parentheses",
        prompt="Write a Python function `valid_parentheses(s: str) -> bool` that determines if the input string containing brackets '(', ')', '{', '}', '[' and ']' is valid. An input is valid if open brackets are closed by the same type of brackets in the correct order.",
        function_name="valid_parentheses",
        unit_tests="""
assert valid_parentheses("()") == True
assert valid_parentheses("()[]{}") == True
assert valid_parentheses("(]") == False
assert valid_parentheses("([)]") == False
assert valid_parentheses("{[]}") == True
assert valid_parentheses("(") == False
"""
    ),
    CodingTask(
        task_id="task_09",
        name="Flatten Nested List",
        prompt="Write a Python function `flatten(nested: list) -> list` that flattens an arbitrarily nested list of integers into a single flat list.",
        function_name="flatten",
        unit_tests="""
assert flatten([1, [2, 3], [[4], 5]]) == [1, 2, 3, 4, 5]
assert flatten([]) == []
assert flatten([[[[1]]]]) == [1]
assert flatten([1, 2, 3]) == [1, 2, 3]
"""
    ),
    CodingTask(
        task_id="task_09_2",
        name="Find Missing Number",
        prompt="Write a Python function `find_missing(nums: list[int]) -> int` that, given an array of n distinct integers in the range [0, n], returns the only number in the range that is missing from the array.",
        function_name="find_missing",
        unit_tests="""
assert find_missing([3, 0, 1]) == 2
assert find_missing([0, 1]) == 2
assert find_missing([9,6,4,2,3,5,7,0,1]) == 8
assert find_missing([0]) == 1
"""
    ),
    CodingTask(
        task_id="task_11",
        name="Caesar Cipher",
        prompt="Write a Python function `caesar_cipher(text: str, shift: int) -> str` that encrypts a string by shifting uppercase and lowercase letters by `shift` positions along the alphabet, wrapping around z/Z. Non-alphabetic characters remain unchanged.",
        function_name="caesar_cipher",
        unit_tests="""
assert caesar_cipher("abc", 3) == "def"
assert caesar_cipher("XYZ", 2) == "ZAB"
assert caesar_cipher("Hello, World!", 5) == "Mjqqt, Btwqi!"
assert caesar_cipher("abc", 0) == "abc"
"""
    ),
    CodingTask(
        task_id="task_12",
        name="Maximum Subarray Sum",
        prompt="Write a Python function `max_subarray(nums: list[int]) -> int` that finds the contiguous subarray with the largest sum and returns its sum.",
        function_name="max_subarray",
        unit_tests="""
assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5, 4, -1, 7, 8]) == 23
assert max_subarray([-1, -2, -3]) == -1
"""
    ),
    CodingTask(
        task_id="task_13",
        name="Matrix Transpose",
        prompt="Write a Python function `transpose(matrix: list[list[int]]) -> list[list[int]]` that returns the transpose of a 2D matrix.",
        function_name="transpose",
        unit_tests="""
assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert transpose([[1]]) == [[1]]
"""
    ),
    CodingTask(
        task_id="task_14",
        name="Run-Length Encoding",
        prompt="Write a Python function `run_length_encode(s: str) -> str` that returns the run-length encoded string representing consecutive character counts (e.g. 'aaabbc' -> 'a3b2c1'). If empty, return ''.",
        function_name="run_length_encode",
        unit_tests="""
assert run_length_encode("aaabbc") == "a3b2c1"
assert run_length_encode("a") == "a1"
assert run_length_encode("") == ""
assert run_length_encode("aabbbcccc") == "a2b3c4"
"""
    ),
    CodingTask(
        task_id="task_15",
        name="Binary Search",
        prompt="Write a Python function `binary_search(nums: list[int], target: int) -> int` that returns the index of target in sorted array nums in O(log n) time, or -1 if not found.",
        function_name="binary_search",
        unit_tests="""
assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
assert binary_search([5], 5) == 0
assert binary_search([], 5) == -1
"""
    ),
    CodingTask(
        task_id="task_16",
        name="Coin Change (DP)",
        prompt="Write a Python function `coin_change(coins: list[int], amount: int) -> int` that returns the fewest number of coins needed to make up `amount`. If that amount of money cannot be made up by any combination of the coins, return -1. You may assume infinite supply of each coin.",
        function_name="coin_change",
        unit_tests="""
assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([2, 5, 10, 1], 27) == 4
"""
    ),
    CodingTask(
        task_id="task_17",
        name="Evaluate Reverse Polish Notation",
        prompt="Write a Python function `eval_rpn(tokens: list[str]) -> int` that evaluates the value of an arithmetic expression in Reverse Polish Notation. Valid operators are '+', '-', '*', and '/'. Division truncates toward zero (e.g. int(a / b)).",
        function_name="eval_rpn",
        unit_tests="""
assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
"""
    ),
    CodingTask(
        task_id="task_18",
        name="Longest Consecutive Sequence",
        prompt="Write a Python function `longest_consecutive(nums: list[int]) -> int` that returns the length of the longest consecutive elements sequence in an unsorted array of integers. Must run in O(n) time.",
        function_name="longest_consecutive",
        unit_tests="""
assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
assert longest_consecutive([]) == 0
assert longest_consecutive([1]) == 1
"""
    ),
    CodingTask(
        task_id="task_19",
        name="Trapping Rain Water",
        prompt="Write a Python function `trap_rain_water(height: list[int]) -> int` that computes how much water an elevation map can trap after raining.",
        function_name="trap_rain_water",
        unit_tests="""
assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap_rain_water([4, 2, 0, 3, 2, 5]) == 9
assert trap_rain_water([]) == 0
assert trap_rain_water([1, 2, 3]) == 0
"""
    ),
    CodingTask(
        task_id="task_20",
        name="Word Break",
        prompt="Write a Python function `word_break(s: str, word_dict: list[str]) -> bool` that returns True if string `s` can be segmented into a space-separated sequence of one or more dictionary words, and False otherwise. Words in the dictionary may be reused multiple times.",
        function_name="word_break",
        unit_tests="""
assert word_break("leetcode", ["leet", "code"]) == True
assert word_break("applepenapple", ["apple", "pen"]) == True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False
assert word_break("", ["a"]) == True
"""
    ),
]

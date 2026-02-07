"""
Dash Evaluation Suite.

Usage:
    python -m dash.evals.run_evals              # String matching (default)
    python -m dash.evals.run_evals --llm-grader # LLM-based grading
    python -m dash.evals.run_evals --compare-results  # Golden SQL comparison
"""

from dash.evals.grader import GradeResult, compare_results, grade_response
from dash.evals.test_cases import CATEGORIES, TEST_CASES, TestCase

__all__ = [
    "TEST_CASES",
    "CATEGORIES",
    "TestCase",
    "grade_response",
    "compare_results",
    "GradeResult",
]

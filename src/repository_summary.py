from src.llm_service import ask_repository


def generate_repository_summary(context):

    question = """
Give a repository overview.

Include:

1. Purpose
2. Main functionality
3. Important files
4. Architecture overview
5. Technologies used

Keep answer concise.
"""

    return ask_repository(
        question,
        context
    )
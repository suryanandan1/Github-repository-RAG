from src.llm_service import ask_repository


def generate_recommendations(
    repository_summary
):

    question = """
Review this repository.

Provide:

1. Architecture improvements
2. Performance improvements
3. Security issues
4. Code quality improvements
5. Scalability suggestions
"""

    return ask_repository(
        question,
        repository_summary
    )
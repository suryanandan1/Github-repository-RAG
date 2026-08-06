import os
import time

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

client = Mistral(
    api_key=os.getenv("MISTRALAI_API_KEY")
)

MODEL_NAME = "codestral-latest"


def _build_prompt(
    question,
    context,
    chat_history
):

    return f"""
You are a senior software engineer.

Repository Context:
{context}

Conversation History:
{chat_history}

Current Question:
{question}

Rules:
1. If the question is about the repository, answer using repository context.
2. Mention filenames, classes, and functions whenever available.
3. If repository information is insufficient, say so.
4. If the question is a general software/programming question unrelated to the repository, answer normally using your knowledge, without relying on repository context.
5. Clearly separate repository-based answers from general knowledge.
6. Do not invent code that is not present in the repository.
7. Use the conversation history to resolve follow-up questions - e.g. "it", "this", "the project" usually refer back to the repository being discussed.
"""


def classify_question(question, chat_history=""):
    """
    Decide whether a question is about the repository currently
    being discussed (including follow-ups that refer back to it,
    e.g. "who created it") or a general software/programming
    knowledge question unrelated to any specific repository.

    Returns "repository" or "general". Defaults to "repository" on
    any classification failure, since repository context rarely
    hurts a general answer (the main prompt's rules already tell
    the model to fall back to general knowledge when context isn't
    relevant) - it's just wasted retrieval, not a correctness risk.
    """

    prompt = f"""
You are a classifier for a repository chat assistant.

Conversation History:
{chat_history}

Question:
{question}

Decide whether this question is about the specific GitHub
repository being discussed - its code, files, structure, purpose,
authorship, dependencies, classes, functions, etc. - including
follow-up questions that refer back to it using words like "it",
"this", "the project", "the repo". If so, reply REPOSITORY.

Otherwise, if it is a general software/programming knowledge
question unrelated to this specific repository (e.g. "What is
Python?", "What is OOP?"), reply GENERAL.

Reply with exactly one word: REPOSITORY or GENERAL. No punctuation,
no explanation.
"""

    try:

        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        if "GENERAL" in answer:
            return "general"

        return "repository"

    except Exception:

        return "repository"


def ask_repository(
    question,
    context,
    chat_history=""
):

    prompt = _build_prompt(
        question,
        context,
        chat_history
    )

    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


def ask_repository_stream(
    question,
    context,
    chat_history=""
):
    """
    Streaming generator version of ask_repository.

    Yields the answer incrementally instead of returning it all at
    once, so callers (e.g. a Streamlit chat UI) can render text as
    it is generated.

    - If Mistral's streaming API is available, chunks are yielded
      directly from the API as they arrive (true token-by-token
      streaming).
    - If streaming is unavailable, or the streaming call fails or
      returns no content, this falls back to fetching the complete
      answer via ask_repository() and yielding it word-by-word with
      a small delay, producing a ChatGPT-like typing effect.
    """

    prompt = _build_prompt(
        question,
        context,
        chat_history
    )

    try:

        stream_response = client.chat.stream(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        streamed_any_content = False

        for event in stream_response:

            delta = None

            try:
                delta = (
                    event.data
                    .choices[0]
                    .delta
                    .content
                )
            except AttributeError:
                delta = None

            if delta:
                streamed_any_content = True
                yield delta

        if not streamed_any_content:
            raise RuntimeError(
                "Mistral stream returned no content."
            )

    except Exception:

        # --------------------------------------------------
        # Fallback: simulate streaming from a full response
        # --------------------------------------------------

        full_answer = ask_repository(
            question,
            context,
            chat_history
        )

        for word in full_answer.split():

            yield word + " "

            time.sleep(0.03)
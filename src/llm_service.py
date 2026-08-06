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

Chat History:
{chat_history}

Repository Context:
{context}

Question:
{question}

Rules:
1. If the question is about the repository, answer using repository context.
2. Mention filenames whenever possible.
3. If repository information is insufficient, say so.
4. If the question is a general software/programming question unrelated to the repository, answer normally using your knowledge.
5. Clearly separate repository-based answers from general knowledge.
6. Do not invent code that is not present in the repository.
"""


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
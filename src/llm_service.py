import os

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

client = Mistral(
    api_key=os.getenv("MISTRALAI_API_KEY")
)


def ask_repository(
    question,
    context,
    chat_history=""
):

    prompt = f"""
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

    response = client.chat.complete(
        model="codestral-latest",
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
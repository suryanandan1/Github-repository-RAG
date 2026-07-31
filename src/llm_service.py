import os

from mistralai import Mistral
from dotenv import load_dotenv
from src.llm_service import ask_repository

load_dotenv()

client = Mistral(
    api_key=os.getenv(
        "MISTRAL_API_KEY"
    )
)


def ask_repository(
    question,
    context
):

    prompt = f"""
You are a senior software engineer.

Repository Context:

{context}

Question:

{question}

Rules:
1. Answer only from repository code.
2. Mention filenames if possible.
3. If information is missing say so.
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
def detect_stack(documents):

    stack = set()

    for doc in documents:

        content = doc["content"]

        if "streamlit" in content:
            stack.add("Streamlit")

        if "fastapi" in content:
            stack.add("FastAPI")

        if "flask" in content:
            stack.add("Flask")

        if "langchain" in content:
            stack.add("LangChain")

        if "chromadb" in content:
            stack.add("ChromaDB")

        if "mistral" in content:
            stack.add("Mistral AI")

    return sorted(stack)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\nclass ",
            "\ndef ",
            "\n\n",
            "\n",
            " "
        ]
    )

    chunks = []

    for doc in documents:

        text_chunks = splitter.split_text(
            doc["content"]
        )

        for chunk in text_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "file_name": doc["file_name"],
                    "file_path": doc["file_path"]
                }
            )

    return chunks
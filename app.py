import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.github_loader import clone_repository

from src.chunker import create_chunks

from src.vector_store import VectorStore

from src.mistral_embeddings import generate_embedding
from src.retriever import retrieve_context

from src.file_reader import (
    get_python_files,
    create_documents
)

from src.repository_analyzer import (
    calculate_repository_stats
)

load_dotenv()

REPO_STORAGE_PATH = os.getenv(
    "REPO_STORAGE_PATH",
    "repos"
)

st.set_page_config(
    page_title="GitHub Repository Assistant",
    layout="wide"
)

st.title(
    "GitHub Repository Assistant"
)

repo_url = st.text_input(
    "Enter GitHub Repository URL"
)

if st.button(
    "Process Repository"
):

    try:

        clone_path = os.path.join(
            REPO_STORAGE_PATH,
            str(uuid.uuid4())
        )

        with st.spinner(
            "Cloning repository..."
        ):
            clone_repository(
                repo_url,
                clone_path
            )

        with st.spinner(
            "Reading files..."
        ):
            files = get_python_files(
                clone_path
            )

            documents = (create_documents(files))
            chunks = create_chunks(documents)

            vector_store = VectorStore()

            progress = st.progress(0)

            total_chunks = len(chunks)

            for index, chunk in enumerate(chunks):

                embedding = generate_embedding(
                    chunk["text"]
                )

                vector_store.add_chunk(
                    chunk_id=f"chunk_{index}",
                    text=chunk["text"],
                    embedding=embedding,
                    metadata={
                        "file_name":
                        chunk["file_name"],

                        "file_path":
                        chunk["file_path"]
                    }
                )

                progress.progress(
                    (index + 1) / total_chunks
                )
        st.success(
            f"""
            Repository Indexed Successfully

            Chunks Created:
            {len(chunks)}

            Chroma Records:
            {vector_store.count()}
            """
        )

        stats = (
            calculate_repository_stats(
                documents
            )
        )

        st.success(
            "Repository processed successfully"
        )

        st.subheader(
            "Repository Statistics"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Python Files",
                stats["total_files"]
            )

            st.metric(
                "Total Lines",
                stats["total_lines"]
            )

        with col2:

            st.metric(
                "Largest File",
                stats["largest_file"]
            )

            st.metric(
                "Largest File Lines",
                stats[
                    "largest_file_lines"
                ]
            )

        st.divider()

        st.subheader(
            "File Preview"
        )

        file_names = [
            doc["file_name"]
            for doc in documents
        ]

        selected_file = st.selectbox(
            "Select File",
            file_names
        )

        selected_doc = next(
            doc
            for doc in documents
            if doc["file_name"]
            == selected_file
        )

        st.code(
            selected_doc["content"][:5000],
            language="python"
        )

    except Exception as e:

        st.error(str(e))

st.divider()

st.header(
    "Repository Chat"
)

question = st.text_input(
    "Ask anything about repository"
)
if question:

    with st.spinner(
        "Searching repository..."
    ):

        context = retrieve_context(
            question
        )

    with st.spinner(
        "Generating answer..."
    ):

        answer = ask_repository(
            question,
            context
        )

    st.markdown(answer)
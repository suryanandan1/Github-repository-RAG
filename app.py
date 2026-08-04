import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.github_loader import clone_repository
from src.chunker import create_chunks
from src.vector_store import VectorStore
from src.mistral_embeddings import generate_embedding
from src.retriever import retrieve_context
from src.llm_service import ask_repository

from src.file_reader import (
    get_python_files,
    create_documents
)

from src.repository_analyzer import (
    calculate_repository_stats
)

from src.repository_summary import (
    generate_repository_summary
)

from src.dependency_analyzer import (
    build_dependency_graph
)

from src.code_reviewer import (
    generate_recommendations
)

from src.tech_stack import detect_stack

from src.repository_health import (
    calculate_health_score
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

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False

if "repo_summary" not in st.session_state:
    st.session_state.repo_summary = ""

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# --------------------------------------------------
# Repository Input
# --------------------------------------------------

repo_url = st.text_input(
    "Enter GitHub Repository URL"
)

# Sidebar

with st.sidebar:

    st.header("Chat History")

    if st.button("New Chat"):
        st.session_state.messages = []

    st.divider()

    for idx, msg in enumerate(
        st.session_state.messages
    ):
        if msg["role"] == "user":
            st.write(
                f"{idx+1}. {msg['content'][:40]}"
            )

    st.divider()

    st.header("Repository")

    if st.session_state.repo_loaded:

        st.success(
            "Repository Loaded"
        )

        st.write(repo_url)

        st.write(
            f"Messages: "
            f"{len(st.session_state.messages)}"
        )

    st.divider()

    st.header("Quick Actions")

    if st.button("Repository Summary"):
        st.session_state.selected_question = (
            "Summarize repository"
        )

    if st.button("Architecture"):
        st.session_state.selected_question = (
            "Explain repository architecture"
        )

    if st.button("Main Classes"):
        st.session_state.selected_question = (
            "List all important classes and explain them"
        )

    if st.button("Main Functions"):
        st.session_state.selected_question = (
            "List all important functions and explain them"
        )

    if st.button("How To Run"):
        st.session_state.selected_question = (
            "Explain how to run this project step by step"
        )

    if st.button("Potential Bugs"):
        st.session_state.selected_question = (
            "Find possible bugs and issues in this repository"
        )

    if st.button("Code Improvements"):
        st.session_state.selected_question = (
            "Suggest improvements for this repository"
        )

# --------------------------------------------------
# Process Repository
# --------------------------------------------------

if st.button(
    "Process Repository"
):

    try:

        st.session_state.messages = []

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

            documents = create_documents(
                files
            )

            chunks = create_chunks(
                documents
            )

            vector_store = VectorStore()

            # Clear previous repository
            vector_store.clear_collection()

            progress = st.progress(0)

            total_chunks = len(
                chunks
            )

            for index, chunk in enumerate(
                chunks
            ):

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
                    (index + 1)
                    / total_chunks
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

        st.session_state.repo_loaded = True

        # --------------------------------------------------
        # Repository Summary
        # --------------------------------------------------

        summary_context = ""

        for chunk in chunks[:20]:

            summary_context += (
                chunk["text"]
                + "\n\n"
            )

        with st.spinner(
            "Generating repository summary..."
        ):

            st.session_state.repo_summary = (
                generate_repository_summary(
                    summary_context
                )
            )

        stats = (
            calculate_repository_stats(
                documents
            )
        )

        st.success(
            "Repository processed successfully"
        )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        st.subheader(
            "Repository Summary"
        )

        st.markdown(
            st.session_state.repo_summary
        )

        st.subheader(
            "Quick Questions"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            if st.button(
                "Explain Architecture"
            ):
                st.session_state.selected_question = (
                    "Explain the architecture of this repository"
                )

        with col2:

            if st.button(
                "Main Files"
            ):
                st.session_state.selected_question = (
                    "What are the most important files in this repository?"
                )

        with col3:

            if st.button(
                "How To Run"
            ):
                st.session_state.selected_question = (
                    "How do I run this project?"
                )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

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
                stats["largest_file_lines"]
            )

        # --------------------------------------------------
        # Tech Stack
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Detected Technology Stack"
        )

        stack = detect_stack(
            documents
        )

        if stack:

            for item in stack:

                st.success(item)

        else:

            st.info(
                "No framework detected"
            )

        # --------------------------------------------------
        # Health Score
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Repository Health Score"
        )

        health_score = (
            calculate_health_score(
                stats
            )
        )

        st.progress(
            health_score / 100
        )

        st.metric(
            "Health Score",
            f"{health_score}/100"
        )

        # --------------------------------------------------
        # Dependency Graph
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Dependency Graph"
        )

        dependency_graph = (
            build_dependency_graph(
                documents
            )
        )

        if dependency_graph:

            for module, imports in (
                dependency_graph.items()
            ):

                st.markdown(
                    f"### {module}"
                )

                if imports:

                    for imp in imports:

                        st.write(
                            f"→ {imp}"
                        )

                else:

                    st.write(
                        "No imports found"
                    )

        # --------------------------------------------------
        # AI Code Review
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "AI Code Review"
        )

        review_context = ""

        for doc in documents[:10]:

            review_context += (
                doc["content"][:2000]
                + "\n\n"
            )

        recommendations = (
            generate_recommendations(
                review_context
            )
        )

        st.markdown(
            recommendations
        )

        # --------------------------------------------------
        # File Preview
        # --------------------------------------------------

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

        with st.expander(
            "View Source Code"
        ):

            st.code(
                selected_doc["content"],
                language="python"
            )

        # st.code(
        #     selected_doc["content"][:5000],
        #     language="python"
        # )

    except Exception as e:

        st.error(str(e))

# --------------------------------------------------
# Chat Section
# --------------------------------------------------

st.divider()

st.header(
    "Repository Chat"
)

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

user_question = st.chat_input(
    "Ask anything about repository..."
)

question = None

if st.session_state.selected_question:

    question = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = ""

elif user_question:

    question = user_question

if (
    question
    and st.session_state.repo_loaded
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    with st.spinner(
        "Searching repository..."
    ):

        context = retrieve_context(
            question
        )

    chat_history = ""

    for msg in (
        st.session_state.messages[-6:]
    ):

        chat_history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    with st.spinner(
        "Generating answer..."
    ):

        answer = ask_repository(
            question,
            context,
            chat_history
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message(
        "assistant"
    ):
        st.markdown(
            answer
        )

elif (
    question
    and not st.session_state.repo_loaded
):

    st.warning(
        "Please process a repository first."
    )
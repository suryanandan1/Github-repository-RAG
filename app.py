import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.github_loader import clone_repository
from src.chunker import create_chunks
from src.vector_store import VectorStore
from src.mistral_embeddings import generate_embedding
from src.retriever import retrieve_context
from src.llm_service import (
    ask_repository,
    ask_repository_stream,
    classify_question
)

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

from src.github_auth import (
    get_login_url,
    exchange_code_for_token,
    get_user_profile
)

from src.github_api import (
    get_user_repositories,
    search_repositories,
    build_authenticated_clone_url
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

if "current_view" not in st.session_state:
    st.session_state.current_view = "dashboard"

if "documents" not in st.session_state:
    st.session_state.documents = []

if "stats" not in st.session_state:
    st.session_state.stats = {}

if "clone_path" not in st.session_state:
    st.session_state.clone_path = ""

if "dependency_graph" not in st.session_state:
    st.session_state.dependency_graph = {}

if "recommendations" not in st.session_state:
    st.session_state.recommendations = ""

if "tech_stack" not in st.session_state:
    st.session_state.tech_stack = []

if "architecture_analysis" not in st.session_state:
    st.session_state.architecture_analysis = ""

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "chroma_count" not in st.session_state:
    st.session_state.chroma_count = 0

if "health_score" not in st.session_state:
    st.session_state.health_score = 0

if "loaded_repo_url" not in st.session_state:
    st.session_state.loaded_repo_url = ""

if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# ---- Phase 6: GitHub OAuth / Explorer state ----

if "github_logged_in" not in st.session_state:
    st.session_state.github_logged_in = False

if "github_user" not in st.session_state:
    st.session_state.github_user = None

if "github_token" not in st.session_state:
    st.session_state.github_token = None

if "selected_repository" not in st.session_state:
    st.session_state.selected_repository = None

if "github_repositories" not in st.session_state:
    st.session_state.github_repositories = []

if "trigger_explorer_load" not in st.session_state:
    st.session_state.trigger_explorer_load = False

# --------------------------------------------------
# GitHub OAuth Callback Handling
# --------------------------------------------------
# When GitHub redirects back after the user authorizes the app, it
# appends a `code` query parameter to GITHUB_REDIRECT_URI. Catch it
# here, exchange it for an access token, and fetch the profile.

query_params = st.query_params

if "code" in query_params and not st.session_state.github_logged_in:

    try:

        auth_code = query_params["code"]

        with st.spinner("Completing GitHub login..."):

            access_token = exchange_code_for_token(
                auth_code
            )

            profile = get_user_profile(
                access_token
            )

        st.session_state.github_token = access_token
        st.session_state.github_user = profile
        st.session_state.github_logged_in = True

        st.query_params.clear()

        st.rerun()

    except Exception as e:

        st.error(
            f"GitHub login failed: {e}"
        )

# --------------------------------------------------
# Shared Repository Processing Pipeline
# --------------------------------------------------
# Reused by both manual URL processing and one-click Repository
# Explorer loading, so the indexing pipeline lives in exactly one
# place.


def process_repository(clone_url, display_url):

    st.session_state.messages = []

    clone_path = os.path.join(
        REPO_STORAGE_PATH,
        str(uuid.uuid4())
    )

    with st.spinner(
        "Cloning repository..."
    ):
        clone_repository(
            clone_url,
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

    # Store documents & stats for sidebar/tool views
    st.session_state.documents = documents
    st.session_state.stats = stats
    st.session_state.clone_path = clone_path
    st.session_state.loaded_repo_url = display_url
    st.session_state.chunks_count = len(chunks)
    st.session_state.chroma_count = vector_store.count()

    # Store tech stack
    st.session_state.tech_stack = detect_stack(
        documents
    )

    # Store dependency graph
    st.session_state.dependency_graph = (
        build_dependency_graph(
            documents
        )
    )

    # Store AI review
    review_context = ""

    for doc in documents[:10]:

        review_context += (
            doc["content"][:2000]
            + "\n\n"
        )

    with st.spinner(
        "Generating code review recommendations..."
    ):

        st.session_state.recommendations = (
            generate_recommendations(
                review_context
            )
        )

    # --------------------------------------------------
    # Architecture Analysis
    # --------------------------------------------------

    with st.spinner(
        "Analyzing repository architecture..."
    ):

        architecture_context = retrieve_context(
            "repository structure and architecture overview"
        )

        st.session_state.architecture_analysis = (
            ask_repository(
                "Provide a detailed architecture analysis of this "
                "repository. Explain its structure, main modules, "
                "how components interact, and the overall design.",
                architecture_context,
                ""
            )
        )

    # --------------------------------------------------
    # Health Score
    # --------------------------------------------------

    st.session_state.health_score = (
        calculate_health_score(
            stats
        )
    )

    st.session_state.repo_loaded = True
    st.session_state.current_view = "dashboard"


# --------------------------------------------------
# Repository Input (Manual Mode - unchanged)
# --------------------------------------------------

repo_url = st.text_input(
    "Enter GitHub Repository URL"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    # ------------------------------
    # GitHub Login
    # ------------------------------

    st.header("GitHub Login")

    if not st.session_state.github_logged_in:

        try:
            login_url = get_login_url()

            st.link_button(
                "🔐 Login with GitHub",
                login_url,
                use_container_width=True
            )

        except Exception as e:
            st.caption(
                f"GitHub login unavailable: {e}"
            )

        st.caption(
            "Log in to browse and load your public, private, "
            "and organization repositories."
        )

    else:

        user = st.session_state.github_user

        col_avatar, col_info = st.columns(
            [1, 2]
        )

        with col_avatar:
            if user.get("avatar"):
                st.image(
                    user["avatar"],
                    width=56
                )

        with col_info:
            st.markdown(
                f"**{user.get('name')}**"
            )
            st.caption(
                f"@{user.get('username')}"
            )

        st.write(
            f"**Followers:** "
            f"{user.get('followers', 0)}"
        )

        st.write(
            f"**Public Repositories:** "
            f"{user.get('public_repos', 0)}"
        )

        if st.button(
            "Logout",
            use_container_width=True
        ):
            st.session_state.github_logged_in = False
            st.session_state.github_user = None
            st.session_state.github_token = None
            st.session_state.selected_repository = None
            st.session_state.github_repositories = []
            st.rerun()

    st.divider()

    # ------------------------------
    # Repository Explorer
    # ------------------------------

    st.header("Repository Explorer")

    if st.session_state.github_logged_in:

        search_query = st.text_input(
            "Search Repository",
            placeholder="e.g. streamlit"
        )

        if st.button(
            "🔄 Refresh Repositories",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Fetching repositories..."
                ):
                    st.session_state.github_repositories = (
                        get_user_repositories(
                            st.session_state.github_token
                        )
                    )
            except Exception as e:
                st.error(str(e))

        if not st.session_state.github_repositories:
            try:
                with st.spinner(
                    "Loading your repositories..."
                ):
                    st.session_state.github_repositories = (
                        get_user_repositories(
                            st.session_state.github_token
                        )
                    )
            except Exception as e:
                st.error(str(e))

        repo_list = st.session_state.github_repositories

        if search_query:
            try:
                repo_list = search_repositories(
                    st.session_state.github_token,
                    search_query
                )
            except Exception as e:
                st.error(str(e))
                repo_list = []

        if repo_list:

            repo_labels = [
                repo["full_name"]
                for repo in repo_list
            ]

            selected_label = st.selectbox(
                "Select Repository",
                repo_labels
            )

            selected_repo = next(
                repo for repo in repo_list
                if repo["full_name"] == selected_label
            )

            st.session_state.selected_repository = selected_repo

            with st.expander(
                "Repository Metadata"
            ):
                st.write(
                    f"**Name:** {selected_repo['name']}"
                )
                st.write(
                    f"**Owner:** {selected_repo['owner']}"
                )
                st.write(
                    f"**Stars:** {selected_repo['stars']}"
                )
                st.write(
                    f"**Forks:** {selected_repo['forks']}"
                )
                st.write(
                    f"**Language:** "
                    f"{selected_repo['language'] or '—'}"
                )
                st.write(
                    f"**Description:** "
                    f"{selected_repo['description'] or '—'}"
                )
                st.write(
                    f"**Last Updated:** "
                    f"{selected_repo['updated_at'] or '—'}"
                )

                if selected_repo["private"]:
                    st.caption("🔒 Private repository")
                else:
                    st.caption("🌐 Public repository")

            if st.button(
                "📥 Load Repository",
                use_container_width=True
            ):
                st.session_state.trigger_explorer_load = True
                st.rerun()

        else:
            st.caption(
                "No repositories found."
            )

    else:
        st.caption(
            "Log in with GitHub to browse and load your "
            "repositories here."
        )

    st.divider()

    # ------------------------------
    # Repository Actions (Tools)
    # ------------------------------

    st.header("Repository Actions")

    tools_disabled = not st.session_state.repo_loaded

    if st.button(
        "Repository Summary",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "summary"
        st.rerun()

    if st.button(
        "File Preview",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "files"
        st.rerun()

    if st.button(
        "Dependency Graph",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "dependency"
        st.rerun()

    if st.button(
        "Tech Stack",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "stack"
        st.rerun()

    if st.button(
        "Code Review",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "review"
        st.rerun()

    if st.button(
        "Architecture Analysis",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "architecture"
        st.rerun()

    st.divider()

    if st.button(
        "🏠 Back to Dashboard",
        use_container_width=True,
        disabled=tools_disabled
    ):
        st.session_state.current_view = "dashboard"
        st.rerun()

    st.divider()

    # ------------------------------
    # Chat History
    # ------------------------------

    st.header("Chat History")

    if st.button("New Chat", use_container_width=True):
        # Clears the conversation only - the cloned repository,
        # its embeddings, and everything in ChromaDB are untouched.
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if st.session_state.messages:

        for idx, msg in enumerate(
            st.session_state.messages
        ):
            if msg["role"] == "user":

                if st.button(
                    f"{idx + 1}. {msg['content'][:40]}",
                    key=f"history_{idx}",
                    use_container_width=True
                ):
                    st.session_state.selected_question = (
                        msg["content"]
                    )
                    st.rerun()
    else:
        st.caption("No previous questions yet.")

    st.divider()

    # ------------------------------
    # Repository Information
    # ------------------------------

    st.header("Repository Information")

    st.write(
        f"**Repository URL:** "
        f"{st.session_state.loaded_repo_url or '—'}"
    )

    st.write(
        f"**Total Messages:** "
        f"{len(st.session_state.messages)}"
    )

    if st.session_state.repo_loaded:
        st.success("Repository Loaded")
    else:
        st.warning("No Repository Loaded")

    st.divider()

    # ------------------------------
    # Quick Actions
    # ------------------------------

    st.header("Quick Actions")

    quick_actions_disabled = not st.session_state.repo_loaded

    if st.button(
        "Summarize Repository",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "Give me a summary of this repository."
        )
        st.rerun()

    if st.button(
        "Explain Architecture",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "Explain the architecture of this repository."
        )
        st.rerun()

    if st.button(
        "List Main Classes",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "List all important classes in this repository "
            "and explain them."
        )
        st.rerun()

    if st.button(
        "List Main Functions",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "List all important functions in this repository "
            "and explain them."
        )
        st.rerun()

    if st.button(
        "Show Dependencies",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "What are the dependencies and imports used in "
            "this repository?"
        )
        st.rerun()

    if st.button(
        "Show Tech Stack",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "What technologies, frameworks, and libraries are "
            "used in this repository?"
        )
        st.rerun()

    if st.button(
        "Show Repository Health",
        use_container_width=True,
        disabled=quick_actions_disabled
    ):
        st.session_state.selected_question = (
            "What is the health status of this repository, and "
            "what could be improved?"
        )
        st.rerun()

# --------------------------------------------------
# Process Repository (Manual URL Mode)
# --------------------------------------------------

if st.button(
    "Process Repository"
):

    try:
        process_repository(
            repo_url,
            repo_url
        )
        st.rerun()

    except Exception as e:

        st.error(str(e))

# --------------------------------------------------
# One-Click Load (Repository Explorer Mode)
# --------------------------------------------------

if st.session_state.trigger_explorer_load:

    st.session_state.trigger_explorer_load = False

    repo = st.session_state.selected_repository

    if repo:

        try:

            clone_url = build_authenticated_clone_url(
                repo["clone_url"],
                st.session_state.github_token
            )

            process_repository(
                clone_url,
                repo["html_url"] or repo["full_name"]
            )

            st.rerun()

        except Exception as e:

            st.error(str(e))

st.divider()

# --------------------------------------------------
# Main Screen Content
# --------------------------------------------------

if not st.session_state.repo_loaded:

    st.info(
        "Paste a GitHub repository URL above, or log in with "
        "GitHub in the sidebar to browse and load your "
        "repositories, then click **Process Repository** / "
        "**Load Repository** to get started."
    )

else:

    current_view = st.session_state.current_view

    # ------------------------------
    # Dashboard (minimal main screen)
    # ------------------------------

    if current_view == "dashboard":

        st.success("Repository Indexed Successfully")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "Chunks Created",
                st.session_state.chunks_count
            )

        with col_b:
            st.metric(
                "Chroma Records",
                st.session_state.chroma_count
            )

        st.success("Repository Processed Successfully")

        st.divider()

        st.subheader("Repository Statistics")

        stats = st.session_state.stats

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Python Files",
                stats.get("total_files", 0)
            )

            st.metric(
                "Total Lines",
                stats.get("total_lines", 0)
            )

        with col2:

            st.metric(
                "Largest File",
                stats.get("largest_file", "—")
            )

            st.metric(
                "Largest File Lines",
                stats.get("largest_file_lines", 0)
            )

        st.divider()

        st.subheader("Repository Health Score")

        st.progress(
            st.session_state.health_score / 100
        )

        st.metric(
            "Health Score",
            f"{st.session_state.health_score}/100"
        )

    # ------------------------------
    # Repository Summary
    # ------------------------------

    elif current_view == "summary":

        st.subheader("Repository Summary")

        st.markdown(
            st.session_state.repo_summary
        )

    # ------------------------------
    # File Preview
    # ------------------------------

    elif current_view == "files":

        st.subheader("File Preview")

        file_names = [
            doc["file_name"]
            for doc in st.session_state.documents
        ]

        selected_file = st.selectbox(
            "Select File",
            file_names
        )

        st.session_state.selected_file = selected_file

        selected_doc = next(
            doc
            for doc in st.session_state.documents
            if doc["file_name"] == selected_file
        )

        st.code(
            selected_doc["content"],
            language="python"
        )

    # ------------------------------
    # Dependency Graph
    # ------------------------------

    elif current_view == "dependency":

        st.subheader("Dependency Graph")

        dependency_graph = st.session_state.dependency_graph

        if dependency_graph:

            for module, imports in dependency_graph.items():

                st.markdown(f"### {module}")

                if imports:
                    for imp in imports:
                        st.write(f"→ {imp}")
                else:
                    st.write("No imports found")
        else:
            st.info("No dependency information available.")

    # ------------------------------
    # Tech Stack
    # ------------------------------

    elif current_view == "stack":

        st.subheader("Technology Stack")

        tech_stack = st.session_state.tech_stack

        if tech_stack:
            for item in tech_stack:
                st.success(item)
        else:
            st.info("No frameworks, libraries, or databases detected.")

    # ------------------------------
    # Code Review
    # ------------------------------

    elif current_view == "review":

        st.subheader("Code Review")

        st.markdown(
            st.session_state.recommendations
        )

    # ------------------------------
    # Architecture Analysis
    # ------------------------------

    elif current_view == "architecture":

        st.subheader("Architecture Analysis")

        st.markdown(
            st.session_state.architecture_analysis
        )

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

    # Last 10 messages (including the question just asked) give
    # the model conversation memory, so follow-ups like "who
    # created it" can resolve "it" back to the repository.
    chat_history = ""

    for msg in (
        st.session_state.messages[-10:]
    ):

        chat_history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    with st.spinner(
        "Understanding your question..."
    ):

        question_type = classify_question(
            question,
            chat_history
        )

    if question_type == "repository":

        with st.spinner(
            "Searching repository..."
        ):

            context = retrieve_context(
                question
            )

    else:

        # General programming/software knowledge question - skip
        # repository retrieval entirely and let the model answer
        # from its own knowledge.
        context = (
            "Not applicable - this is a general knowledge "
            "question unrelated to the repository."
        )

    with st.chat_message(
        "assistant"
    ):

        placeholder = st.empty()

        full_response = ""

        for chunk in ask_repository_stream(
            question,
            context,
            chat_history
        ):

            full_response += chunk

            placeholder.markdown(
                full_response + "▌"
            )

        placeholder.markdown(
            full_response
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

elif (
    question
    and not st.session_state.repo_loaded
):

    st.warning(
        "Please process a repository first."
    )
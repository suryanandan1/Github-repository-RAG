import os
import re
import uuid

import pandas as pd
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

from src.commit_service import (
    get_recent_commits,
    get_commit_analytics,
    summarize_commits,
    CommitServiceError
)

from src.github_commits import (
    get_commits,
    get_commit_details,
    GitHubCommitsError
)

from src.github_activity import (
    get_developer_activity,
    get_commit_timeline,
    filter_commits_by_days,
    get_last_updated_status,
    is_activity_question,
    build_activity_context,
    generate_activity_summary,
    GitHubActivityError
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

# ---- Commit Intelligence state ----

if "commits" not in st.session_state:
    st.session_state.commits = []

if "commit_summary" not in st.session_state:
    st.session_state.commit_summary = ""

if "commit_stats" not in st.session_state:
    st.session_state.commit_stats = {}

# ---- Repository Activity Intelligence state (Phase 6.5) ----

if "activity_commits" not in st.session_state:
    st.session_state.activity_commits = []

if "activity_summary_text" not in st.session_state:
    st.session_state.activity_summary_text = ""

if "commit_search_query" not in st.session_state:
    st.session_state.commit_search_query = ""

if "timeline_range_days" not in st.session_state:
    st.session_state.timeline_range_days = 30


def extract_repo_full_name(url):
    """
    Best-effort extraction of "owner/repo" from a GitHub URL, so
    Commit Intelligence works in manual URL mode too, not just
    when a repository was picked via the Repository Explorer.
    """

    if not url:
        return None

    match = re.search(
        r"github\.com[/:]([^/]+)/([^/.\s]+)",
        url
    )

    if not match:
        return None

    owner, name = match.group(1), match.group(2)

    return f"{owner}/{name}"

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

    # Reset Commit Intelligence state - independent feature, but
    # commit data from a previously loaded repo shouldn't persist
    # once a different repository is processed.
    st.session_state.commits = []
    st.session_state.commit_stats = {}
    st.session_state.commit_summary = ""

    # Repository Activity Intelligence - independent feature, same
    # reasoning: don't let activity data from a previous repo
    # linger once a different repository is processed.
    st.session_state.activity_commits = []
    st.session_state.activity_summary_text = ""
    st.session_state.commit_search_query = ""

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

    # ------------------------------
    # Commit Intelligence
    # ------------------------------
    # Lives in the Repository Explorer section, works in both
    # login and manual URL modes. Nothing runs automatically -
    # commit history is only fetched, analyzed, or summarized
    # when one of these buttons is explicitly clicked. Results
    # render on the main screen once available.

    st.markdown("**Commit Intelligence**")

    active_repo_name = None

    if st.session_state.selected_repository:
        active_repo_name = (
            st.session_state.selected_repository["full_name"]
        )
    elif st.session_state.loaded_repo_url:
        active_repo_name = extract_repo_full_name(
            st.session_state.loaded_repo_url
        )

    if st.button(
        "📜 Load Commit History",
        use_container_width=True,
        disabled=not st.session_state.repo_loaded
    ):

        try:

            with st.spinner("Loading commits..."):

                commits = get_recent_commits(
                    active_repo_name,
                    limit=50,
                    access_token=st.session_state.github_token
                )

            st.session_state.commits = commits
            st.session_state.commit_stats = {}
            st.session_state.commit_summary = ""

            if not commits:
                st.info(
                    "No commits were found for this repository."
                )

            st.rerun()

        except CommitServiceError as e:
            st.error(str(e))

        except Exception as e:
            st.error(
                f"Unexpected error loading commits: {e}"
            )

    if st.session_state.commits:

        if st.button(
            "📊 Commit Analytics",
            use_container_width=True
        ):

            try:
                st.session_state.commit_stats = (
                    get_commit_analytics(
                        st.session_state.commits
                    )
                )
                st.rerun()

            except Exception as e:
                st.error(
                    f"Failed to compute commit analytics: {e}"
                )

        if st.button(
            "🤖 Commit Summary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Generating commit summary..."
                ):
                    st.session_state.commit_summary = (
                        summarize_commits(
                            st.session_state.commits
                        )
                    )

                st.rerun()

            except CommitServiceError as e:
                st.error(str(e))

            except Exception as e:
                st.error(
                    f"Unexpected error generating summary: {e}"
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
    # Repository Activity (Phase 6.5)
    # ------------------------------

    st.header("Repository Activity")

    activity_disabled = not st.session_state.repo_loaded

    if st.button(
        "Commit History",
        use_container_width=True,
        disabled=activity_disabled
    ):
        try:
            with st.spinner("Fetching commit history..."):
                st.session_state.activity_commits = get_commits(
                    st.session_state.github_token,
                    active_repo_name,
                    limit=50
                )
            st.session_state.current_view = "activity_history"
            st.rerun()
        except GitHubCommitsError as e:
            st.error(str(e))

    if st.button(
        "Developer Activity",
        use_container_width=True,
        disabled=activity_disabled
    ):
        try:
            if not st.session_state.activity_commits:
                with st.spinner("Fetching commit history..."):
                    st.session_state.activity_commits = get_commits(
                        st.session_state.github_token,
                        active_repo_name,
                        limit=100
                    )
            st.session_state.current_view = "activity_developer"
            st.rerun()
        except GitHubCommitsError as e:
            st.error(str(e))

    if st.button(
        "Commit Timeline",
        use_container_width=True,
        disabled=activity_disabled
    ):
        try:
            if not st.session_state.activity_commits:
                with st.spinner("Fetching commit history..."):
                    st.session_state.activity_commits = get_commits(
                        st.session_state.github_token,
                        active_repo_name,
                        limit=100
                    )
            st.session_state.current_view = "activity_timeline"
            st.rerun()
        except GitHubCommitsError as e:
            st.error(str(e))

    if st.button(
        "Recent Changes",
        use_container_width=True,
        disabled=activity_disabled
    ):
        try:
            if not st.session_state.activity_commits:
                with st.spinner("Fetching commit history..."):
                    st.session_state.activity_commits = get_commits(
                        st.session_state.github_token,
                        active_repo_name,
                        limit=20
                    )
            st.session_state.current_view = "activity_recent_changes"
            st.rerun()
        except GitHubCommitsError as e:
            st.error(str(e))

    if st.button(
        "Activity Summary",
        use_container_width=True,
        disabled=activity_disabled
    ):
        try:
            if not st.session_state.activity_commits:
                with st.spinner("Fetching commit history..."):
                    st.session_state.activity_commits = get_commits(
                        st.session_state.github_token,
                        active_repo_name,
                        limit=100
                    )
            with st.spinner("Generating activity summary..."):
                st.session_state.activity_summary_text = (
                    generate_activity_summary(
                        st.session_state.activity_commits,
                        days=30
                    )
                )
            st.session_state.current_view = "activity_summary"
            st.rerun()
        except (GitHubCommitsError, GitHubActivityError) as e:
            st.error(str(e))

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

    # ------------------------------
    # Repository Activity: Commit History
    # ------------------------------

    elif current_view == "activity_history":

        st.subheader("Commit History")

        search_col, clear_col = st.columns([4, 1])

        with search_col:
            search_query = st.text_input(
                "Search commits (author, message, or SHA)",
                value=st.session_state.commit_search_query
            )

        with clear_col:
            st.write("")
            if st.button("Clear Search", use_container_width=True):
                st.session_state.commit_search_query = ""
                st.rerun()

        st.session_state.commit_search_query = search_query

        commits_to_show = st.session_state.activity_commits

        if search_query:

            query_lower = search_query.strip().lower()

            commits_to_show = [
                c for c in commits_to_show
                if query_lower in c["message"].lower()
                or query_lower in c["sha"].lower()
                or query_lower in c["author_name"].lower()
            ]

        if not commits_to_show:
            st.info(
                "No commits match your search."
                if search_query else
                "No commits found."
            )

        for commit in commits_to_show:

            with st.expander(
                f"{commit['message_title']}  ·  {commit['short_sha']}"
            ):

                avatar_col, info_col = st.columns([1, 4])

                with avatar_col:
                    if commit.get("author_avatar"):
                        st.image(
                            commit["author_avatar"],
                            width=48
                        )

                with info_col:

                    st.write(
                        f"**Author:** {commit['author_name']}"
                    )

                    st.write(
                        f"**Date:** "
                        f"{commit['date'][:10] if commit['date'] else '—'}"
                    )

                    st.write(
                        f"**SHA:** `{commit['sha']}`"
                    )

                    if commit.get("branch"):
                        st.write(
                            f"**Branch:** {commit['branch']}"
                        )

                    if commit.get("url"):
                        st.markdown(
                            f"[View on GitHub]({commit['url']})"
                        )

    # ------------------------------
    # Repository Activity: Developer Activity
    # ------------------------------

    elif current_view == "activity_developer":

        st.subheader("Developer Activity")

        developer_activity = get_developer_activity(
            st.session_state.activity_commits
        )

        if not developer_activity:
            st.info("No commit data available.")

        else:

            rows = [
                {
                    "Contributor": dev["name"],
                    "Commits": dev["commits"],
                    "Last Commit": (
                        dev["last_commit_date"] or ""
                    )[:10]
                }
                for dev in developer_activity
            ]

            st.dataframe(
                rows,
                use_container_width=True
            )

            st.bar_chart(
                data=pd.Series(
                    {
                        dev["name"]: dev["commits"]
                        for dev in developer_activity
                    },
                    name="Commits"
                )
            )

            st.success(
                f"Most Active Contributor: "
                f"{developer_activity[0]['name']}"
            )

    # ------------------------------
    # Repository Activity: Commit Timeline
    # ------------------------------

    elif current_view == "activity_timeline":

        st.subheader("Commit Timeline")

        range_options = [7, 30, 90]

        range_choice = st.radio(
            "Time Range",
            options=range_options,
            format_func=lambda d: f"{d} days",
            horizontal=True,
            index=range_options.index(
                st.session_state.timeline_range_days
            )
        )

        st.session_state.timeline_range_days = range_choice

        filtered_commits = filter_commits_by_days(
            st.session_state.activity_commits,
            range_choice
        )

        if not filtered_commits:
            st.info(
                f"No commits in the last {range_choice} days."
            )

        else:

            st.metric(
                "Commits in Range",
                len(filtered_commits)
            )

            st.markdown("**Commits per Day**")
            daily = get_commit_timeline(
                filtered_commits,
                granularity="day"
            )
            st.bar_chart(
                pd.Series(daily, name="Commits")
            )

            st.markdown("**Commits per Week**")
            weekly = get_commit_timeline(
                filtered_commits,
                granularity="week"
            )
            st.bar_chart(
                pd.Series(weekly, name="Commits")
            )

            st.markdown("**Commits per Month**")
            monthly = get_commit_timeline(
                filtered_commits,
                granularity="month"
            )
            st.bar_chart(
                pd.Series(monthly, name="Commits")
            )

    # ------------------------------
    # Repository Activity: Recent Changes
    # ------------------------------

    elif current_view == "activity_recent_changes":

        st.subheader("Recent Changes")

        if not st.session_state.activity_commits:
            st.info("No recent commits available.")

        else:

            for commit in st.session_state.activity_commits[:10]:

                with st.expander(
                    f"{commit['message_title']}  ·  "
                    f"{commit['short_sha']}"
                ):

                    st.write(
                        f"**Author:** {commit['author_name']}"
                    )

                    st.write(
                        f"**Date:** "
                        f"{commit['date'][:10] if commit['date'] else '—'}"
                    )

                    st.write(
                        f"**Full SHA:** `{commit['sha']}`"
                    )

                    detail_key = f"details_result_{commit['sha']}"

                    if st.button(
                        "View Changed Files",
                        key=f"details_btn_{commit['sha']}"
                    ):
                        try:
                            with st.spinner(
                                "Fetching commit details..."
                            ):
                                st.session_state[detail_key] = (
                                    get_commit_details(
                                        st.session_state.github_token,
                                        active_repo_name,
                                        commit["sha"]
                                    )
                                )
                        except GitHubCommitsError as e:
                            st.error(str(e))

                    details = st.session_state.get(detail_key)

                    if details:

                        st.write(
                            f"**Additions:** +{details['additions']}"
                        )

                        st.write(
                            f"**Deletions:** -{details['deletions']}"
                        )

                        st.write("**Files Changed:**")

                        for f in details["files"]:
                            st.write(
                                f"- {f['filename']} "
                                f"({f['status']}, "
                                f"+{f['additions']}/-{f['deletions']})"
                            )

    # ------------------------------
    # Repository Activity: Activity Summary
    # ------------------------------

    elif current_view == "activity_summary":

        st.subheader("Activity Summary")

        status = get_last_updated_status(
            st.session_state.activity_commits
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Last Commit",
                (status["last_commit_date"] or "—")[:10]
                if status["last_commit_date"] else "—"
            )

        with col2:
            st.metric(
                "Days Since Last Commit",
                status["days_since_last_commit"]
                if status["days_since_last_commit"] is not None
                else "—"
            )

        with col3:
            st.metric(
                "Activity Status",
                status["status"]
            )

        st.divider()

        if st.session_state.activity_summary_text:
            st.markdown(
                st.session_state.activity_summary_text
            )
        else:
            st.info(
                "Click 'Activity Summary' in the sidebar to "
                "generate one."
            )

# --------------------------------------------------
# Commit History (Commit Intelligence)
# --------------------------------------------------
# Shown only once commit history has actually been loaded - never
# fetched automatically.

if st.session_state.commits:

    st.divider()

    st.header("Commit History")

    commit_rows = [
        {
            "SHA": commit["sha"],
            "Author": commit["author"],
            "Message": commit["message"],
            "Date": commit["date"]
        }
        for commit in st.session_state.commits
    ]

    st.dataframe(
        commit_rows,
        use_container_width=True
    )

    # ------------------------------
    # Commit Analytics
    # ------------------------------

    if st.session_state.commit_stats:

        st.subheader("Commit Analytics")

        stats = st.session_state.commit_stats

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Commits",
                stats.get("total_commits", 0)
            )

        with col2:
            st.metric(
                "Most Active Contributor",
                stats.get("most_active_contributor") or "—"
            )

        top_contributors = stats.get(
            "top_contributors", []
        )

        if top_contributors:

            st.markdown("**Top Contributors**")

            contributor_rows = [
                {
                    "Contributor": name,
                    "Commits": count
                }
                for name, count in top_contributors
            ]

            st.dataframe(
                contributor_rows,
                use_container_width=True
            )

            st.bar_chart(
                data=pd.Series(
                    dict(top_contributors),
                    name="Commits"
                )
            )

    # ------------------------------
    # Commit Summary
    # ------------------------------

    if st.session_state.commit_summary:

        st.subheader("Commit Summary")

        st.markdown(
            st.session_state.commit_summary
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

        if is_activity_question(question, chat_history):
            question_type = "activity"
        else:
            question_type = classify_question(
                question,
                chat_history
            )

    if question_type == "activity":

        # Commit/contributor/recent-activity questions use live
        # GitHub Commit API data instead of ChromaDB retrieval.
        try:

            with st.spinner(
                "Checking repository activity..."
            ):

                activity_commits_for_chat = (
                    st.session_state.activity_commits
                    or get_commits(
                        st.session_state.github_token,
                        active_repo_name,
                        limit=50
                    )
                )

                st.session_state.activity_commits = (
                    activity_commits_for_chat
                )

                developer_activity = get_developer_activity(
                    activity_commits_for_chat
                )

                activity_status = get_last_updated_status(
                    activity_commits_for_chat
                )

                context = build_activity_context(
                    activity_commits_for_chat,
                    developer_activity,
                    activity_status
                )

        except GitHubCommitsError as e:

            context = (
                f"Commit activity data could not be retrieved: {e}"
            )

    elif question_type == "repository":

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
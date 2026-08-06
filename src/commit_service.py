import os
from collections import Counter

import requests
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"

# Kept independent from src/llm_service.py on purpose - Commit
# Intelligence should not depend on (or risk destabilizing) the
# existing RAG chat client/prompting.
_COMMIT_MODEL_NAME = "codestral-latest"
_commit_client = None


class CommitServiceError(Exception):
    """
    Raised for any user-facing Commit Intelligence failure (missing
    repository, GitHub API error, rate limiting, network issues,
    empty history, etc.) so the UI layer can show a friendly
    st.error() message instead of a raw traceback.
    """
    pass


def _get_commit_client():

    global _commit_client

    if _commit_client is None:

        _commit_client = Mistral(
            api_key=os.getenv("MISTRALAI_API_KEY")
        )

    return _commit_client


class CommitService:
    """
    Thin wrapper around the GitHub REST API for fetching commit
    history. Works with or without an access token - unauthenticated
    requests work for public repos (subject to GitHub's lower
    unauthenticated rate limit), while a token is required for
    private repositories and raises the standard GitHub rate limit.
    """

    def __init__(self, access_token=None):
        self.access_token = access_token

    def _headers(self):

        headers = {
            "Accept": "application/vnd.github+json"
        }

        if self.access_token:
            headers["Authorization"] = (
                f"Bearer {self.access_token}"
            )

        return headers

    def get_recent_commits(self, repo_name, limit=50):
        """
        Fetch the most recent `limit` commits for `repo_name`
        ("owner/repo"). Returns a list of dicts:

        {"sha", "author", "message", "date", "url"}

        Raises CommitServiceError with a friendly message on any
        failure (missing repo, auth/rate-limit issues, network
        problems, malformed response).
        """

        if not repo_name or "/" not in repo_name:
            raise CommitServiceError(
                "No repository selected. Load a repository first."
            )

        try:

            response = requests.get(
                f"{GITHUB_API_BASE}/repos/{repo_name}/commits",
                headers=self._headers(),
                params={
                    "per_page": min(max(limit, 1), 100)
                },
                timeout=10
            )

        except requests.exceptions.RequestException as e:

            raise CommitServiceError(
                f"Network error while fetching commits: {e}"
            )

        if response.status_code == 404:
            raise CommitServiceError(
                f"Repository '{repo_name}' was not found, or you "
                f"don't have access to it."
            )

        if response.status_code == 401:
            raise CommitServiceError(
                "GitHub authentication failed. Please log in again."
            )

        if response.status_code == 403:

            remaining = response.headers.get(
                "X-RateLimit-Remaining"
            )

            if remaining == "0":
                raise CommitServiceError(
                    "GitHub API rate limit exceeded. Log in with "
                    "GitHub for a higher limit, or try again later."
                )

            raise CommitServiceError(
                "Access denied. This may be a private repository "
                "you don't have permission to view."
            )

        if response.status_code == 409:
            # Repository exists but has no commits yet (empty repo)
            return []

        if not response.ok:
            raise CommitServiceError(
                f"GitHub API error ({response.status_code}): "
                f"{response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError:
            raise CommitServiceError(
                "Received an invalid response from GitHub."
            )

        if not isinstance(data, list):
            raise CommitServiceError(
                "Unexpected response format from GitHub."
            )

        commits = []

        for item in data[:limit]:

            commit_info = item.get("commit", {}) or {}
            author_info = commit_info.get("author", {}) or {}
            gh_author = item.get("author") or {}

            author_name = (
                gh_author.get("login")
                or author_info.get("name")
                or "Unknown"
            )

            raw_message = commit_info.get("message") or ""

            commits.append(
                {
                    "sha": (item.get("sha") or "")[:7],
                    "author": author_name,
                    "message": raw_message.split("\n")[0],
                    "date": author_info.get("date") or "",
                    "url": item.get("html_url") or ""
                }
            )

        return commits


# --------------------------------------------------
# Standalone helper functions
# --------------------------------------------------
# Business logic lives here, kept separate from the Streamlit UI
# layer in app.py, which only calls these functions and renders
# their results.


def get_recent_commits(repo_name, limit=50, access_token=None):
    """
    Convenience wrapper around CommitService.get_recent_commits so
    callers don't need to instantiate the class themselves.
    """

    service = CommitService(
        access_token=access_token
    )

    return service.get_recent_commits(
        repo_name,
        limit=limit
    )


def get_commit_analytics(commits):
    """
    Compute summary analytics from a list of commit dicts:
    total commit count, per-contributor commit counts (sorted,
    most active first), and the single most active contributor.
    """

    if not commits:
        return {
            "total_commits": 0,
            "contributor_counts": {},
            "top_contributors": [],
            "most_active_contributor": None
        }

    authors = [
        commit.get("author", "Unknown")
        for commit in commits
    ]

    counts = Counter(authors)

    top_contributors = counts.most_common()

    return {
        "total_commits": len(commits),
        "contributor_counts": dict(counts),
        "top_contributors": top_contributors,
        "most_active_contributor": (
            top_contributors[0][0]
            if top_contributors else None
        )
    }


def build_commit_context(commits):
    """
    Build a plain "Author | Date | Message" block from commit
    dicts, used as the context fed to the LLM for summarization.
    """

    lines = []

    for commit in commits:

        date = (commit.get("date") or "")[:10]

        lines.append(
            f"{commit.get('author', 'Unknown')} | "
            f"{date} | "
            f"{commit.get('message', '')}"
        )

    return "\n".join(lines)


def summarize_commits(commits):
    """
    Generate a professional development summary from commit
    history using Mistral. Raises CommitServiceError if there is
    no commit history to summarize, or if the LLM call fails.
    """

    if not commits:
        raise CommitServiceError(
            "No commit history available to summarize. "
            "Load commit history first."
        )

    commit_context = build_commit_context(
        commits
    )

    prompt = f"""
You are a GitHub repository analyst.

Analyze the following commit history and provide:

1. Recent development summary
2. Major features added
3. Bugs fixed
4. Development trends
5. Contributor activity insights

Commit History:
{commit_context}

Generate concise, professional output.
"""

    try:

        client = _get_commit_client()

        response = client.chat.complete(
            model=_COMMIT_MODEL_NAME,
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

    except Exception as e:

        raise CommitServiceError(
            f"Failed to generate commit summary: {e}"
        )
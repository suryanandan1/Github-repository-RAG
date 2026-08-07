import requests

GITHUB_API_BASE = "https://api.github.com"


class GitHubCommitsError(Exception):
    """
    Raised for any user-facing GitHub Commits API failure (missing
    repository, auth/rate-limit issues, network errors, malformed
    responses) so the UI layer can show a friendly message.
    """
    pass


def _headers(access_token):

    headers = {
        "Accept": "application/vnd.github+json"
    }

    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    return headers


def _handle_response_errors(response, repo_name):

    if response.status_code == 404:
        raise GitHubCommitsError(
            f"Repository '{repo_name}' was not found, or you "
            f"don't have access to it."
        )

    if response.status_code == 401:
        raise GitHubCommitsError(
            "GitHub authentication failed. Please log in again."
        )

    if response.status_code == 403:

        remaining = response.headers.get(
            "X-RateLimit-Remaining"
        )

        if remaining == "0":
            raise GitHubCommitsError(
                "GitHub API rate limit exceeded. Log in with "
                "GitHub for a higher limit, or try again later."
            )

        raise GitHubCommitsError(
            "Access denied. This may be a private repository "
            "you don't have permission to view."
        )

    if response.status_code == 409:
        # Empty repository (no commits yet) - not an error
        return

    if not response.ok:
        raise GitHubCommitsError(
            f"GitHub API error ({response.status_code}): "
            f"{response.text[:200]}"
        )


def _format_commit(item, default_branch=None):

    commit_info = item.get("commit", {}) or {}
    author_info = commit_info.get("author", {}) or {}
    gh_author = item.get("author") or {}

    raw_message = commit_info.get("message") or ""

    return {
        "sha": item.get("sha") or "",
        "short_sha": (item.get("sha") or "")[:7],
        "message": raw_message,
        "message_title": raw_message.split("\n")[0],
        "author_name": (
            gh_author.get("login")
            or author_info.get("name")
            or "Unknown"
        ),
        "author_avatar": gh_author.get("avatar_url"),
        "date": author_info.get("date") or "",
        "url": item.get("html_url") or "",
        # GitHub's commit-list endpoint doesn't report which
        # branch(es) a commit belongs to without expensive extra
        # calls per commit, so this is a best-effort fallback to
        # the repository's default branch.
        "branch": default_branch
    }


def get_repository_default_branch(access_token, repo_name):

    try:

        response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo_name}",
            headers=_headers(access_token),
            timeout=10
        )

        if response.ok:
            return response.json().get("default_branch")

    except requests.exceptions.RequestException:
        pass

    return None


def get_commits(
    access_token,
    repo_name,
    limit=50,
    author=None,
    since=None,
    until=None
):
    """
    Fetch commits for repo_name ("owner/repo"), most recent first.

    `since` / `until` are ISO 8601 strings, e.g.
    "2026-07-01T00:00:00Z" - passed straight through to GitHub's
    API, which filters server-side.
    """

    if not repo_name or "/" not in repo_name:
        raise GitHubCommitsError(
            "No repository selected. Load a repository first."
        )

    default_branch = get_repository_default_branch(
        access_token,
        repo_name
    )

    commits = []
    page = 1
    per_page = min(limit, 100)

    try:

        while len(commits) < limit:

            params = {
                "per_page": per_page,
                "page": page
            }

            if author:
                params["author"] = author

            if since:
                params["since"] = since

            if until:
                params["until"] = until

            response = requests.get(
                f"{GITHUB_API_BASE}/repos/{repo_name}/commits",
                headers=_headers(access_token),
                params=params,
                timeout=10
            )

            if response.status_code == 409:
                break

            _handle_response_errors(
                response,
                repo_name
            )

            batch = response.json()

            if not isinstance(batch, list) or not batch:
                break

            commits.extend(
                _format_commit(item, default_branch)
                for item in batch
            )

            if len(batch) < per_page:
                break

            page += 1

    except requests.exceptions.RequestException as e:

        raise GitHubCommitsError(
            f"Network error while fetching commits: {e}"
        )

    return commits[:limit]


def get_commit_details(access_token, repo_name, sha):
    """
    Fetch full details for a single commit: message, full SHA,
    author, commit time, stats (additions/deletions/total), and
    the list of changed files.
    """

    if not repo_name or "/" not in repo_name:
        raise GitHubCommitsError(
            "No repository selected."
        )

    try:

        response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo_name}/commits/{sha}",
            headers=_headers(access_token),
            timeout=15
        )

    except requests.exceptions.RequestException as e:

        raise GitHubCommitsError(
            f"Network error while fetching commit details: {e}"
        )

    _handle_response_errors(
        response,
        repo_name
    )

    data = response.json()

    commit_info = data.get("commit", {}) or {}
    author_info = commit_info.get("author", {}) or {}
    gh_author = data.get("author") or {}
    stats = data.get("stats", {}) or {}

    files = [
        {
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0)
        }
        for f in (data.get("files", []) or [])
    ]

    return {
        "sha": data.get("sha"),
        "message": commit_info.get("message") or "",
        "author_name": (
            gh_author.get("login")
            or author_info.get("name")
            or "Unknown"
        ),
        "date": author_info.get("date") or "",
        "additions": stats.get("additions", 0),
        "deletions": stats.get("deletions", 0),
        "total_changes": stats.get("total", 0),
        "files": files,
        "url": data.get("html_url") or ""
    }


def search_commits(
    access_token,
    repo_name,
    query=None,
    author=None,
    since=None,
    until=None,
    limit=100
):
    """
    Search commits by author and/or date range (filtered
    server-side by GitHub's API), then by free-text query matched
    against message, SHA, or author name (filtered client-side,
    since GitHub's commit-list endpoint has no message search).
    """

    commits = get_commits(
        access_token,
        repo_name,
        limit=limit,
        author=author,
        since=since,
        until=until
    )

    if not query:
        return commits

    query_lower = query.strip().lower()

    return [
        commit for commit in commits
        if query_lower in commit["message"].lower()
        or query_lower in commit["sha"].lower()
        or query_lower in commit["author_name"].lower()
    ]
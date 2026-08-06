import requests

GITHUB_API_BASE = "https://api.github.com"


def _headers(access_token):

    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json"
    }


def _format_repo(repo):

    owner = repo.get("owner") or {}

    return {
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner": owner.get("login"),
        "description": repo.get("description"),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "language": repo.get("language"),
        "updated_at": repo.get("updated_at"),
        "private": repo.get("private", False),
        "clone_url": repo.get("clone_url"),
        "html_url": repo.get("html_url"),
        "default_branch": repo.get("default_branch")
    }


def get_user_repositories(access_token, per_page=100):
    """
    List repositories the authenticated user can access - repos
    they own, repos they collaborate on, and repos belonging to
    organizations they are a member of. Includes private repos,
    since the "repo" OAuth scope was requested at login.
    """

    repositories = []
    page = 1

    while True:

        response = requests.get(
            f"{GITHUB_API_BASE}/user/repos",
            headers=_headers(access_token),
            params={
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "affiliation": (
                    "owner,collaborator,organization_member"
                )
            },
            timeout=10
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        repositories.extend(batch)

        if len(batch) < per_page:
            break

        page += 1

    return [
        _format_repo(repo)
        for repo in repositories
    ]


def search_repositories(access_token, query, per_page=20):
    """
    Search the authenticated user's accessible repositories by
    name or description.

    GitHub's public search API does not reliably surface private
    repositories, so this filters the user's own repository list
    client-side instead - this keeps private and organization
    repos searchable too.
    """

    if not query:
        return []

    all_repos = get_user_repositories(
        access_token,
        per_page=per_page
        if per_page > 100 else 100
    )

    query_lower = query.strip().lower()

    matches = [
        repo for repo in all_repos
        if query_lower in (repo["name"] or "").lower()
        or query_lower in (repo["description"] or "").lower()
        or query_lower in (repo["full_name"] or "").lower()
    ]

    return matches[:per_page]


def get_repository_details(access_token, owner, repo_name):
    """
    Fetch full metadata for a single repository by owner/name.
    """

    response = requests.get(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}",
        headers=_headers(access_token),
        timeout=10
    )

    response.raise_for_status()

    return _format_repo(
        response.json()
    )


def build_authenticated_clone_url(clone_url, access_token):
    """
    Inject the access token into an HTTPS clone URL so that
    `git clone` can authenticate - required for private and
    organization repositories, and harmless for public ones.
    """

    if not access_token or not clone_url:
        return clone_url

    if clone_url.startswith("https://"):
        return clone_url.replace(
            "https://",
            f"https://{access_token}@",
            1
        )

    return clone_url